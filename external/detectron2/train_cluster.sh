#!/bin/bash
#SBATCH --job-name=train_crow_detectron2
#SBATCH --output=train_nn_%A.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --mem=36G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

module purge
module load Python/3.7.4-GCCcore-8.3.0
module load CUDA
module load cuDNN
module load Anaconda3/5.0.1
source /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate detectron2-env

nvidia-smi

echo "Running detectron2 training..."
python train_crow.py --config-file  --config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --num-gpus 3 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.0025

echo "done"
