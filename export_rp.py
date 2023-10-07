from numpy import require
import torch
import argparse

parser = argparse.ArgumentParser('export the region prompts from a tuned model checkpoint', add_help=False)
parser.add_argument('--model_path', default='logs/checkpoint.pth', type=str)
parser.add_argument('--name', required=True, type=str)
args = parser.parse_args()

model = torch.load(args.model_path, map_location='cpu')['model']

pe_dict = dict()
fc_dict = dict()

for k, v in model.items():
    if 'position' in k and 'attnpool' in k:
        assert k.startswith("backbone.0")
        # pe_dict[k.replace("backbone.0.", "")] = v.clone()
        pe_dict["positional_embedding"] = v.clone()
    if 'fc_cls' in k and 'token' not in k:
        # import ipdb;ipdb.set_trace()
        fc_dict[k.replace("fc_cls.", "")] = v.clone()

# torch.save(pe_dict, f"{args.name}.pth")
# torch.save(pe_dict, f"t5_positional_embedding.pth")
# torch.save(fc_dict, f"t5_fc.pth")
torch.save(pe_dict, f"token_fc/t5_positional_embedding.pth")
torch.save(fc_dict, f"token_fc/t5_fc.pth")

