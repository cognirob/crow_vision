#!/bin/bash
#SBATCH --job-name=train_crow_detectron2
#SBATCH --output=train_nn_%A.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=0:10:00
#SBATCH --partition=gpu

module purge
module load gcccuda/2020
#module load cuDNN
module load Anaconda3/5.3.0
source /opt/apps/software/Anaconda3/5.3.0/etc/profile.d/conda.sh

CROW_VISION_REPO="~/crow_vision"
DETECTRON_REPO="~/detectron2"

nvidia-smi
echo "Installing.."
cd ~
conda env create -f ${CROW_VISION_REPO}/external/detectron2/environment.yml
conda env update -f ${CROW_VISION_REPO}/external/detectron2/environment.yml
conda activate detectron2-env

conda info
module list
python --version
conda list|grep torch
python -c 'import torch; print(torch.cuda.is_available())' || exit 1

python -m pip install --user ${DETECTRON_REPO}

echo "Running detectron2 training..."
#python detectron2/train_crow.py 
python ${DETECTRON_REPO}/tools/train_net.py --config-file  --config-file ${CROW_VISION_REPO}/external/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.0025 
nvidia-smi 

echo "done"
