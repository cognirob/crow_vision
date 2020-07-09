## Run detectron training on custom dataset 
#
# Run as: 
# nohup python train_crow.py --config-file configs/COCO-InstanceSegmentation/coco_instance_segm_mask_rcnn_R_50_FNP_3x_kuka.yaml --resume &
import os
from detectron2.data.datasets import load_coco_json, register_coco_instances

DATASET="/nfs/projects/crow/data/yolact/datasets/dataset_kuka_env_pybullet_fixedcolor/"
DETECTRON_REPO=os.path.expanduser("~/detectron2")

register_coco_instances(
    "kuka_train",
    {},
    DATASET+"/train/annotations.json",
    DATASET+"/train",
)
register_coco_instances(
    "kuka_val",
    {},
    DATASET+"/test/annotations.json",
    DATASET+"/test",
)

exec(open(DETECTRON_REPO+"/tools/train_net.py").read(), globals())  # TODO better pass args

# from predictor import VisualizationDemo
# exec(open("./demo/demo.py").read(), globals()) #TODO better pass args
