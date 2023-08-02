# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import copy
import os
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import inverse, nn, Tensor
from .attention import MultiheadAttention
import torchvision
from models.ops.modules import MSDeformAttn

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class Transformer(nn.Module):

    def __init__(self, args):
        d_model = args.hidden_dim
        dropout = args.dropout
        nhead = args.nheads
        num_queries = args.num_queries
        dim_feedforward = args.dim_feedforward
        num_encoder_layers = args.enc_layers
        num_decoder_layers = args.dec_layers
        num_t2vencoder_layers = args.t2venc_layers
        normalize_before = False
        return_intermediate_dec = True
        query_dim = 4
        activation = 'prelu'
        num_patterns = 0
        bbox_embed_diff_each_layer = True
        stage1_box = args.stage1_box
        keep_query_pos = False
        query_scale_type = 'cond_elewise'
        modulate_hw_attn = True
        use_deformable_attention = args.use_deformable_attention
        self.disable_spatial_attn_mask = args.disable_spatial_attn_mask

        super().__init__()
        self.args = args
        # text proposal to foreground value cross attention encoder
        if self.args.t2v_encoder:
            # import ipdb;ipdb.set_trace()
            t2v_encoder_layer = T2V_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.t2v_encoder = TransformerEncoder(t2v_encoder_layer, num_t2vencoder_layers, encoder_norm, t2v_encoder=True)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, use_deformable_attention=use_deformable_attention)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, mix_encoder=args.mix_encoder)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos, use_deformable_attention=use_deformable_attention and not args.only_deform_enc)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
                                          use_deformable_attention=use_deformable_attention and not args.only_deform_enc, fix_reference_points=args.fix_reference_points)

        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

        self.stage1_box_embed = None
        self.stage1_obj_embed = None
        self.stage1_box = stage1_box

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # code adapted from Deformable-DETR
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, size):
        ori_memory = memory.unflatten(0, size)
        ori_mask = memory_padding_mask.unflatten(1, size)
        inds_h = (~ori_mask).float().cumsum(dim=-2)
        inds_w = (~ori_mask).float().cumsum(dim=-1)
        # normalize to 0, 1
        inds_h = (inds_h - 0.5) / inds_h[:,-1:,:]
        inds_w = (inds_w - 0.5) / inds_w[:,:,-1:]
        center = torch.stack([inds_w, inds_h], dim=-1)
        # pred
        rpn_preds = self.stage1_box_embed(ori_memory)
        rpn_logits = self.stage1_obj_embed(ori_memory)
        rpn_preds = rpn_preds.permute(2,0,1,3)
        rpn_logits = rpn_logits.permute(2,0,1,3)
        pred_center = rpn_preds[:,:,:,:2].sigmoid() - 0.5
        # add center prior
        pred_center_with_offset = pred_center + center
        pred_with_offset = torch.cat([inverse_sigmoid(pred_center_with_offset), rpn_preds[:,:,:,2:]], dim=-1)
        # query embed is NB4
        proposals = pred_with_offset.flatten(1,2).permute(1,0,2)
        logits = rpn_logits.flatten(1,2).permute(1,0,2)

        return proposals, logits

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, src, mask, refpoint_embed, pos_embed, tgt_mask=None, src_query=None, cls_func=None,
        key_content=None, key_position=None, value_binary=None):
        # flatten NxCxHxW to HWxNxC
        assert len(src) == 1 and len(mask) == 1 and len(pos_embed) == 1
        src = src[0]
        mask = mask[0]
        pos_embed = pos_embed[0]
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        valid_ratio = torch.stack([self.get_valid_ratio(m) for m in [mask]], 1)
        mask = mask.flatten(1)
        shape = [(h, w)]
        shape = torch.as_tensor(shape, dtype=torch.long, device=mask.device)
        if self.disable_spatial_attn_mask:
            valid_ratio = None
            shape = None

        if self.args.t2v_encoder:
            image_length = src.shape[0]# [4150, 2, 256] -> 4150
            src = torch.cat((src, key_content))# [4150, 2, 256] [240, 2, 256] -> [4390, 2, 256]
            src = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, key_position=key_position, 
                image_length=image_length, value_binary=value_binary)  # (L, batch_size, d)
            src = src[:image_length]
            memory = src
            mask = mask[:, :image_length]
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, shape=shape, valid_ratio=valid_ratio,
                                key_content=key_content, key_position=key_position, value_binary=value_binary)# ->[4150, 2, 256]

        if refpoint_embed is not None:# [1000, 2, 4] train
            if refpoint_embed.dim() == 2:
                refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
            else:
                assert refpoint_embed.dim() == 3
            classes = None
            out_dict = None
            confidences = None
            if self.args.t2v_rpn:
                # firstly forward the rpn
                proposals, logits = self.gen_encoder_output_proposals(memory, mask, size=(h, w))
                # create a dict
                out_dict = {'pred_logits': logits.permute(1,0,2), 'pred_boxes': proposals.permute(1,0,2).sigmoid()}
        else:
            assert tgt_mask is None
            assert src_query is None
            assert cls_func is not None
            # firstly forward the rpn
            proposals, logits = self.gen_encoder_output_proposals(memory, mask, size=(h, w))
            # create a dict
            out_dict = {'pred_logits': logits.permute(1,0,2), 'pred_boxes': proposals.permute(1,0,2).sigmoid()}
            # select the topk proposals
            selected_inds = logits.topk(k=self.stage1_box, dim=0)[1][:,:,0]
            refpoint_embed = torch.gather(proposals, dim=0, index=selected_inds.unsqueeze(-1).expand(self.stage1_box, bs, 4))
            classes, src_query, refpoint_embed, confidences, key_content, key_position, value_binary = cls_func(refpoint_embed.detach())
            

        # in case of split_class
        if src.size(1) != refpoint_embed.size(1):# split class: [3700, 2, 256] [1000, 4, 4]
            mask = mask.repeat(2, 1)# [2, 3700]->[4, 3700]
            pos_embed = pos_embed.repeat(1, 2, 1)# [3700, 2, 256]->[3700, 4, 256]
            memory = memory.repeat(1, 2, 1)# [3700, 2, 256]->[3700, 4, 256]

        # query_embed = gen_sineembed_for_position(refpoint_embed)
        num_queries = refpoint_embed.shape[0]# 1000
        if self.num_patterns == 0:
            if src_query is not None:# train
                tgt = src_query.permute(1,0,2)
                if self.args.binary_token:
                    tgt = value_binary[0].repeat(num_queries, 1, 1)
                    tgt = tgt + src_query.permute(1,0,2)
            else:
                if self.args.binary_token:
                    tgt = value_binary[0].repeat(num_queries, 1, 1)# TODO:用detach吗？
                else:
                    tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)# [1000, 2, 256]
                if src.size(1) != refpoint_embed.size(1):# split class: [3700, 2, 256] [1000, 4, 4]
                    tgt = tgt.repeat(1, 2, 1)# [1000, 2, 256]->[1000, 4, 256]
        else:
            tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1) # n_q*n_pat, bs, d_model
            refpoint_embed = refpoint_embed.repeat(self.num_patterns, 1, 1) # n_q*n_pat, bs, d_model
        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, refpoints_unsigmoid=refpoint_embed, tgt_mask=tgt_mask, shape=shape, valid_ratio=valid_ratio)
        return hs, references, memory, classes, out_dict, confidences

class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,# [4390, 2, 256]
                src_mask: Optional[Tensor] = None,# None
                src_key_padding_mask: Optional[Tensor] = None,# [2, 4150]
                pos: Optional[Tensor] = None,# [4390, 2, 256]
                image_length=None,# 4150
                value_binary=None,
                ):
        assert image_length is not None
        
        pos_src = self.with_pos_embed(src, pos)# -> [4390, 2, 256]
        q, k, v = pos_src[0:image_length], pos_src[image_length:], value_binary# [4150, 2, 256] [240, 2, 256] [240, 2, 256]
        
        qmask, kmask = src_key_padding_mask, torch.ones_like(src_key_padding_mask[:, :value_binary.shape[0]], dtype=torch.bool)# [2, 4150] [2, 240]
        # attn_mask = torch.matmul(qmask.unsqueeze(2).float(), kmask.unsqueeze(1).float()).bool().repeat(self.nhead, 1, 1)# [16, 4150, 240] TODO: Why?
        src2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=None)[0]# ->[4150, 2, 256]
        src2 = src[:image_length] + self.dropout1(src2)# [4150, 2, 256]
        src3 = self.norm1(src2)# [4150, 2, 256]
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))# [4150, 2, 256]
        src2 = src2 + self.dropout2(src3)# [4150, 2, 256]
        src2 = self.norm2(src2)# [4150, 2, 256]
        src = torch.cat([src2, src[image_length:]])# ->[4390, 2, 256]
        # print('after src shape :',src.shape)
        return src


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256, mix_encoder=False, t2v_encoder=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm
        self.mix_encoder = mix_encoder
        self.t2v_encoder = t2v_encoder
        if self.mix_encoder:
            dim_feedforward = 1024
            dropout = 0.1
            self.nhead = 8
            self.d_model = d_model
            self.key_query_scale = MLP(d_model, d_model, d_model, 2)
            self.ca_qcontent_proj = nn.Linear(d_model, d_model)
            self.ca_qpos_proj = nn.Linear(d_model, d_model)
            self.ca_kcontent_proj = nn.Linear(d_model, d_model)
            self.ca_kpos_proj = nn.Linear(d_model, d_model)
            self.ca_v_proj = nn.Linear(d_model, d_model)
            self.ca_kpos_sine_proj = nn.Linear(d_model, d_model)
            self.cross_attn = MultiheadAttention(d_model * 2, self.nhead, dropout=0.1, vdim=d_model)
            self.ref_point_head = MLP(4 // 2 * d_model, d_model, d_model, 2)
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)

            self.activation = _get_activation_fn("relu")

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                shape = None,
                valid_ratio=None,
                key_content=None, key_position=None, value_binary=None, image_length=None):
        output = src

        if self.t2v_encoder:
            key_points = key_position.sigmoid()# [200, 2, 4]
            obj_center = key_points[..., :4]     # [200, 2, 4] self.query_dim=4
            # get sine embedding for the query vector
            key_sine_embed = gen_sineembed_for_position(obj_center)  # [200, 2, 4]->[200, 2, 512]
            key_sine_embed = key_sine_embed[...,:256]
            # import ipdb;ipdb.set_trace()
            pos = torch.cat((pos, key_sine_embed))
        for layer_id, layer in enumerate(self.layers):
            if self.mix_encoder:
                # image tokens & class tokens cross attention(CA) with fg/bg value
                # tgt/query: output query_pos: pos*pos_scales
                # memory/key: key_content=[key_pos, key_neg] pos: key_position=[key_pos_pos, key_neg_pos]
                # value: [value_fg, value_bg] 
                key_points = key_position.sigmoid()# [200, 2, 4]
                obj_center = key_points[..., :4]     # [200, 2, 4] self.query_dim=4
                # get sine embedding for the query vector
                key_sine_embed = gen_sineembed_for_position(obj_center)  # [200, 2, 4]->[200, 2, 512]
                # key_pos = self.ref_point_head(key_sine_embed) # [200, 2, 512] -> [200, 2, 256] xywh

                # modulated HW attentions
                refHW_cond = self.ref_anchor_head(key_content).sigmoid() # 256->2 [200, 2, 256]->nq, bs, 2 [200, 2, 2] // split class [1000, 4, 256]->[1000, 4, 2]
                key_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)# [200, 2, 128]*[200, 2, 1]
                # refHW_cond: [1000, 2, 2]; obj_center: [1000, 2, 4]
                key_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)# [200, 2, 128]
                key_sine_embed = key_sine_embed[...,:self.d_model]
                # For the first decoder layer, we do not apply transformation over p_s
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.key_query_scale(key_content)# 256->256 [200, 2, 256]->[200, 2, 256]

                # apply transformation
                key_sine_embed = key_sine_embed[...,:self.d_model] * pos_transformation# [200, 2, 512] -> [200, 2, 256] xy
            # rescale the content and pos sim
            pos_scales = self.query_scale(output)
            if self.t2v_encoder:
                output = layer(output, src_mask=mask,
                            src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales, image_length=image_length, value_binary=value_binary)# ->[4150, 2, 256]
            else:
                output = layer(output, src_mask=mask,
                            src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales, shape=shape, valid_ratio=valid_ratio)# ->[4150, 2, 256]
            #为什么这个pos要x pos_scales

            if self.mix_encoder:
                # ========== Begin of Cross-Attention =============
                # Apply projections here
                # shape: num_queries x batch_size x 256
                q_content = self.ca_qcontent_proj(output)# [4150, 2, 256]->[4150, 2, 256]
                k_content = self.ca_kcontent_proj(key_content)# [200, 2, 256]->[200, 2, 256]
                v = self.ca_v_proj(value_binary)# [200, 2, 256]->[200, 2, 256]
                num_queries, bs, n_model = k_content.shape# 200, 2, 256
                hw, _, _ = q_content.shape# 4150 TODO:暂时删掉了split class

                # k_pos = self.ca_kpos_proj(key_pos)# [200, 2, 256]->[200, 2, 256]
                q_pos = self.ca_qpos_proj(pos*pos_scales)# [4150, 2, 256][4150, 2, 256]->[4150, 2, 256]
                # For the first decoder layer, we concatenate the positional embedding predicted from 
                # the object query (the positional embedding) into the original query (key) in DETR.
                # if layer_id == 0:
                #     q = q_content + q_pos
                #     k = k_content + k_pos
                # else:
                q = q_content# [4150, 2, 256]
                k = k_content# [200, 2, 256]
                k = k.view(num_queries, bs, self.nhead, n_model//self.nhead)# [200, 2, 8, 32]
                key_sine_embed = self.ca_kpos_sine_proj(key_sine_embed)# [200, 2, 256]->[200, 2, 256]
                key_sine_embed = key_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)# [200, 2, 256]->[200, 2, 8, 32]
                k = torch.cat([k, key_sine_embed], dim=3).view(num_queries, bs, n_model * 2)# ->[200, 2, 512]
                q = q.view(hw, bs, self.nhead, n_model//self.nhead)# [4150, 2, 256]->[4150, 2, 8, 32]
                q_pos = q_pos.view(hw, bs, self.nhead, n_model//self.nhead)# [4150, 2, 256]->[4150, 2, 8, 32]
                q = torch.cat([q, q_pos], dim=3).view(hw, bs, n_model * 2)# ->[4150, 2, 512]
                output2 = self.cross_attn(query=q,# [4150, 2, 512]# DEGUG之前是output2
                                        key=k,# [200, 2, 512]
                                        value=v, attn_mask=None,# [200, 2, 256] [2*8, 4150, 200]
                                        # value=v, attn_mask=src_key_padding_mask.unsqueeze(-1).repeat(1, 1, len(k)).repeat_interleave(8, dim=0),# [200, 2, 256] [2*8, 4150, 200]
                                        key_padding_mask=None)[0] # ->[4150, 2, 256]          
                # ========== End of Cross-Attention =============
                output = output + self.dropout2(output2)# [4150, 2, 256]
                output = self.norm2(output)
                output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
                output = output + self.dropout3(output2)
                output = self.norm3(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, 
                    d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                    modulate_hw_attn=False,
                    bbox_embed_diff_each_layer=False,
                    use_deformable_attention=False,
                    fix_reference_points=False,
                    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.use_deformable_attention = use_deformable_attention
        assert return_intermediate
        self.query_dim = query_dim
        self.fix_reference_points = fix_reference_points

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        if not keep_query_pos:# False
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,# [1000, 2, 256] [4150, 2, 256]
                tgt_mask: Optional[Tensor] = None,# None
                memory_mask: Optional[Tensor] = None,# None
                tgt_key_padding_mask: Optional[Tensor] = None,# None
                memory_key_padding_mask: Optional[Tensor] = None,# [2, 4150]
                pos: Optional[Tensor] = None,# [4150, 2, 256]
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2 [1000, 2, 4]
                shape=None,# [1, 2] [[50, 83]]
                valid_ratio=None,# [2, 1, 2]
                ):
        # import ipdb;ipdb.set_trace()
        output = tgt# [1000, 2, 256]

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()# [1000, 2, 4]
        if self.fix_reference_points:# False
            reference_points = reference_points * valid_ratio.permute(1,0,2).repeat(1,reference_points.size(1) // valid_ratio.size(0),2)
        
        ref_points = [reference_points]# list[tensor]

        # import ipdb;ipdb.set_trace()
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]     # [num_queries, batch_size, 2] [1000, 2, 4] self.query_dim=4
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)  # [1000, 2, 4]->[1000, 2, 512]
            query_pos = self.ref_point_head(query_sine_embed) # -> [1000, 2, 256] xywh

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':# 'cond_elewise'
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)# 256->256 [1000, 2, 256]->[1000, 2, 256]
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation# [1000, 2, 256] xy

            # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # 256->2 [1000, 2, 256]->nq, bs, 2 [1000, 2, 2] // split class [1000, 4, 256]->[1000, 4, 2]
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)# [1000, 2, 128]*[1000, 2, 1]
                # refHW_cond: [1000, 2, 2]; obj_center: [1000, 2, 4]
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)# [1000, 2, 128]


            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0),
                           reference_points=reference_points[:,:,:2].permute(1,0,2)[:,:,None],
                           shape=shape)

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:# True
                    tmp = self.bbox_embed[layer_id](output)# [1000, 2, 256]->[1000, 2, 4]
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)# [1000, 2, 4]
                new_reference_points = tmp[..., :self.query_dim].sigmoid()# [1000, 2, 4]
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()# [1000, 2, 4]

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2), 
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_deformable_attention=False):
        super().__init__()
        self.use_deformable_attention = use_deformable_attention
        if use_deformable_attention:
            self.self_attn = MSDeformAttn(d_model, 1, nhead, n_points=4)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     shape=None,
                     valid_ratio=None):
        if self.use_deformable_attention:
            reference_points = self.get_reference_points(shape, valid_ratio, device=src.device)
            src2 = self.self_attn(self.with_pos_embed(src, pos).permute(1,0,2), reference_points, src.permute(1,0,2), shape, torch.zeros([1], dtype=int, device=shape.device), src_key_padding_mask)
            src2 = src2.permute(1,0,2)
        else:
            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False, use_deformable_attention=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.use_deformable_attention = use_deformable_attention
        if self.use_deformable_attention:
            self.cross_attn = MSDeformAttn(d_model, 1, nhead, n_points=4, query_dim=2*d_model)
            self.ca_qpos_proj = nn.Linear(d_model, d_model)
        else:
            self.ca_qcontent_proj = nn.Linear(d_model, d_model)
            self.ca_qpos_proj = nn.Linear(d_model, d_model)
            self.ca_kcontent_proj = nn.Linear(d_model, d_model)
            self.ca_kpos_proj = nn.Linear(d_model, d_model)
            self.ca_v_proj = nn.Linear(d_model, d_model)
            self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
            self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False,
                     reference_points=None,
                     shape=None):
        # import ipdb;ipdb.set_trace()
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        if self.use_deformable_attention:
            if is_first or self.keep_query_pos:
                q_pos = self.ca_qpos_proj(query_pos)
                q = tgt + q_pos
            num_queries, bs, n_model = q.shape
            q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
            query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
            tgt2 = self.cross_attn(q.permute(1,0,2), reference_points.contiguous(), memory.permute(1,0,2), shape, torch.zeros([1], dtype=int, device=shape.device), memory_key_padding_mask)
            tgt2 = tgt2.permute(1,0,2)
        else:
            q_content = self.ca_qcontent_proj(tgt)# [1000, 2, 256]
            k_content = self.ca_kcontent_proj(memory)# [4150, 2, 256]
            v = self.ca_v_proj(memory)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            k_pos = self.ca_kpos_proj(pos)  

            # For the first decoder layer, we concatenate the positional embedding predicted from 
            # the object query (the positional embedding) into the original query (key) in DETR.
            if is_first or self.keep_query_pos:
                q_pos = self.ca_qpos_proj(query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content

            q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
            query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
            query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2 = self.cross_attn(query=q,
                                    key=k,
                                    value=v, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(args)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if mask == 'auto':
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError("tensors dim must be 3 or 4 but {}({})".format(self.tensors.dim(), self.tensors.shape))

    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(tensor.dim())
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self):
        """remove the padding and convert to img list
        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {
            'tensors.shape': self.tensors.shape,
            'mask.shape': self.mask.shape
        }

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)