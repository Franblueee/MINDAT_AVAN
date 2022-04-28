#!/bin/bash
#SBATCH --job-name train              # Nombre del proceso
#SBATCH --partition dgx,dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH --mem=80GB                           # Numero de gpus a usar
#SBATCH --out=train_w.txt

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
#conda activate /mnt/homeGPU/fcastro/conda-envs/LULC
conda activate /mnt/homeGPU/fcastro/conda-envs/newlulc
export LD_LIBRARY_PATH=/mnt/homeGPU/fcastro/conda-envs/newlulc/lib


python code/train_w.py --model_name=mobilenetv3large_v3-2 --DA_name=DA4 --ft_mode=1 --batch_size=8
python code/train_w.py --model_name=mobilenetv3large_v3-3 --DA_name=DA4 --ft_mode=1 --batch_size=8
python code/train_w.py --model_name=mobilenetv3large_v1 --DA_name=DA4 --ft_mode=1 --batch_size=8

python code/train_w.py --model_name=mobilenetv3large_v3-2 --DA_name=DA5 --ft_mode=1 --batch_size=8
python code/train_w.py --model_name=mobilenetv3large_v3-3 --DA_name=DA5 --ft_mode=1 --batch_size=8
python code/train_w.py --model_name=mobilenetv3large_v1 --DA_name=DA5 --ft_mode=1 --batch_size=8