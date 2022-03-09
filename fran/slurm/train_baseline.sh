#!/bin/bash
#SBATCH --job-name baseline              # Nombre del proceso
#SBATCH --partition dios   # Cola para ejecutar
# AAAAAAAAAA SBATCH -w hera  
#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH --mem=50GB                           # Numero de gpus a usar
#SBATCH --out=train_baseline.txt

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/fcastro/conda-envs/XAIFL

python code/train_baseline.py
