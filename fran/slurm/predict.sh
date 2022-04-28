#!/bin/bash
#SBATCH --job-name predict              # Nombre del proceso
#SBATCH --partition dgx,dios,dgx2   # Cola para ejecutar
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH --mem=50GB                           # Numero de gpus a usar
#SBATCH --out=predict.txt

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/fcastro/conda-envs/XAIFL

python code/predict.py --model_name=resnet152_baseline --DA_name=DA1 --ft_layers=0
python code/predict.py --model_name=resnet152_baseline --DA_name=DA2 --ft_layers=0

