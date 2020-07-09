## Run detectron training on custom dataset 
#
# Run as: 
# nohup python train_crow.py --config-file configs/COCO-InstanceSegmentation/coco_instance_segm_mask_rcnn_R_50_FNP_3x_kuka.yaml --resume &
import os

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances

#DATASET="/nfs/projects/crow/data/yolact/datasets/dataset_kuka_env_pybullet_fixedcolor"
DATASET=os.path.expanduser("~/crow_vision_yolact/data/yolact/datasets/dataset_kuka_env_pybullet_fixedcolor")
DETECTRON_REPO=os.path.expanduser("~/detectron2")
CONFIG="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
NUM_GPUS=1

# first, register custom COCO-like dataset
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

from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(CONFIG))
cfg.DATASETS.TRAIN = ("kuka_train",)
cfg.DATASETS.TEST = ("kuka_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8*NUM_GPUS
cfg.SOLVER.BASE_LR = 0.00025*NUM_GPUS  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
