#!/bin/bash
#SBATCH --job-name train              # Nombre del proceso
#SBATCH --partition dgx,dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH --mem=50GB                           # Numero de gpus a usar
#SBATCH --out=train_mobilenet.txt

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
#conda activate /mnt/homeGPU/fcastro/conda-envs/LULC
conda activate /mnt/homeGPU/fcastro/conda-envs/newlulc
export LD_LIBRARY_PATH=/mnt/homeGPU/fcastro/conda-envs/newlulc/lib


python code/train.py --model_name=mobilenetv3large_baseline --DA_name=DA4 --ft_mode=1 --batch_size=32
python code/train.py --model_name=mobilenetv3large_v0 --DA_name=DA4 --ft_mode=1 --batch_size=32
python code/train.py --model_name=mobilenetv3large_v1 --DA_name=DA4 --ft_mode=1 --batch_size=32

python code/train.py --model_name=mobilenetv3large_v2-1 --DA_name=DA4 --ft_mode=1 --batch_size=32
python code/train.py --model_name=mobilenetv3large_v2-2 --DA_name=DA4 --ft_mode=1 --batch_size=32
python code/train.py --model_name=mobilenetv3large_v2-3 --DA_name=DA4 --ft_mode=1 --batch_size=32

python code/train.py --model_name=mobilenetv3large_v2-1 --DA_name=DA4 --ft_mode=1 --batch_size=32
python code/train.py --model_name=mobilenetv3large_v2-2 --DA_name=DA4 --ft_mode=1 --batch_size=32
python code/train.py --model_name=mobilenetv3large_v2-3 --DA_name=DA4 --ft_mode=1 --batch_size=32

python code/train.py --model_name=mobilenetv3large_v3-1 --DA_name=DA4 --ft_mode=1 --batch_size=32
python code/train.py --model_name=mobilenetv3large_v3-2 --DA_name=DA4 --ft_mode=1 --batch_size=32
python code/train.py --model_name=mobilenetv3large_v3-3 --DA_name=DA4 --ft_mode=1 --batch_size=32