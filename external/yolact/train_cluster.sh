#!/bin/bash
#SBATCH --job-name=yolact_train_job
#SBATCH --output=yolact_job.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --mem=36G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

## To be run by sbatch on CIIRC cluster
# you need to create conda env yolact-env beforehand. 
YOLACT_REPO='./crow_vision_yolact'
DATASET='fixedcolor'

module purge
module load CUDA/10.1.105-GCC-8.2.0-2.31.1
module load cuDNN
module load Anaconda3/5.0.1
. /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate yolact-env

nvidia-smi
echo "YOLACT trainig started!"
python ${YOLACT_REPO}/train.py --batch_size=2 --dataset_number=${DATASET}
echo "done"
