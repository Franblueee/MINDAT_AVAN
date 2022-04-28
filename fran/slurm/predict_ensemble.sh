#!/bin/bash
#SBATCH --job-name train              # Nombre del proceso
#SBATCH --partition dgx,dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH --mem=30GB                           # Numero de gpus a usar
#SBATCH --out=predict_ensemble.txt

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
#conda activate /mnt/homeGPU/fcastro/conda-envs/LULC
conda activate /mnt/homeGPU/fcastro/conda-envs/newlulc
export LD_LIBRARY_PATH=/mnt/homeGPU/fcastro/conda-envs/newlulc/lib


python code/predict_ensemble.py