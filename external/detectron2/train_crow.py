## Run detectron training on custom dataset 
#
# Run as: 
# nohup python train_crow.py --config-file configs/COCO-InstanceSegmentation/coco_instance_segm_mask_rcnn_R_50_FNP_3x_kuka.yaml --resume &

from detectron2.data.datasets import load_coco_json, register_coco_instances

DATASET="dataset_kuka_env_pybullet_19"

register_coco_instances(
    "kuka_train",
    {},
    "./datasets/"+DATASET+"/train/annotations.json",
    "./datasets/"+DATASET+"/train",
)
register_coco_instances(
    "kuka_val",
    {},
    "./datasets/"+DATASET+"/test/annotations.json",
    "./datasets/"+DATASET+"/test",
)

exec(open("./tools/train_net.py").read(), globals())  # TODO better pass args

# from predictor import VisualizationDemo
# exec(open("./demo/demo.py").read(), globals()) #TODO better pass args
