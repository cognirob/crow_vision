#!/bin/bash
#SBATCH --job-name=train_crow_detectron2
#SBATCH --output=train_nn_%A.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --mem=36G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

module purge
module load CUDA/9.2.88-GCC-7.3.0-2.30
module load cuDNN
module load Anaconda3/5.0.1
source /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate detectron2-env

nvidia-smi

echo "Running detectron2 training..."
python train_crow.py --config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
echo "done"
