#!/bin/bash
#SBATCH --job-name=train_crow_detectron2
#SBATCH --output=train_nn_%A.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --partition=gpu

module purge
module load gcccuda/2020
#module load cuDNN
module load Anaconda3/5.0.1

CROW_VISION_REPO="/home/$USER/crow_vision"

echo "Setup conda env"
source /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
cd ~
#conda env remove -n detectron2-env
conda env create -f ${CROW_VISION_REPO}/external/detectron2/environment.yml || conda env update -f ${CROW_VISION_REPO}/external/detectron2/environment.yml
conda activate detectron2-env
#python -m pip install --user detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/index.html


echo "debug:"
nvidia-smi
conda info
module list
python --version
conda list|grep torch
python -c 'import torch.cuda; print(torch.cuda.is_available())' || exit 1


echo "Running detectron2 training..."
python ${CROW_VISION_REPO}/external/detectron2/train_net.py \
  --config-file "${CROW_VISION_REPO}/external/detectron2/configs/COCO-InstanceSegmentation/crow.yaml" \
  --dataset "/nfs/projects/crow/data/yolact/datasets/dataset_kuka_env_pybullet_fixedcolor/" \
  --num-gpus 4

echo "done"
