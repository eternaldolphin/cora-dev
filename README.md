### align clip roi features into pseudo words in t5 space 
```shell
# COCO RN50
bash configs/instructor-embedding/COCO_RN50_T5_vg.sh exp_name1 4 local
bash configs/instructor-embedding/COCO_RN50_T5_vg_base.sh exp_name2 4 local --resume logs/COCO_RN50_T5_vg/vg/checkpoint0004.pth

#工程TODO: COCO RN50x4 
bash configs/instructor-embedding/COCO_RN50x4_T5_vg.sh exp_name1 4 local
bash configs/instructor-embedding/COCO_RN50x4_T5_vg_base.sh exp_name2 4 local --resume logs/COCO_RN50_T5_vg/vg/checkpoint0004.pth
```


### Region Prompting
```shell
# run the following commands for region prompting
# COCO RN50
bash configs/COCO_RN50.sh exp_name 4 local
# COCO RN50x4
bash configs/COCO_RN50x4.sh exp_name 4 local
# LVIS RN50x4
bash configs/LVIS_RN50x4.sh exp_name 4 local
```
Note that you can also run it on a cluster with slurm by replacing local with slurm in the command.

### Exporting the region prompts
```shell
python export_rp.py --model_path /path/to/trained/model.pth --name output_name
```
