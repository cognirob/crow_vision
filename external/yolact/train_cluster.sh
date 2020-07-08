#!/bin/bash
#SBATCH --job-name=yolact_train_job
#SBATCH --output=yolact_job.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --partition=gpu

## To be run by sbatch on CIIRC cluster
# you need to create conda env yolact-env beforehand. 

module purge
module load CUDA/10.1.105-GCC-8.2.0-2.31.1
module load cuDNN
module load Anaconda3/5.0.1
nvidia-smi
. /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate yolact-env
python ./crow_vision_yolact/train.py --batch_size=2 --dataset_number=fixedcolor

