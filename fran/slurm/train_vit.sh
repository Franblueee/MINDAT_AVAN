#!/bin/bash
#SBATCH --job-name train              # Nombre del proceso
#SBATCH --partition dgx,dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH --mem=50GB                           # Numero de gpus a usar
#SBATCH --out=train_vit.txt

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
#conda activate /mnt/homeGPU/fcastro/conda-envs/LULC
conda activate /mnt/homeGPU/fcastro/conda-envs/newlulc
export LD_LIBRARY_PATH=/mnt/homeGPU/fcastro/conda-envs/newlulc/lib

python code/train.py --model_name=VITs16_baseline --DA_name=DA0 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v0 --DA_name=DA0 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v1 --DA_name=DA0 --ft_mode=0 --batch_size=8

python code/train.py --model_name=VITs16_v2-1 --DA_name=DA0 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v2-2 --DA_name=DA0 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v2-3 --DA_name=DA0 --ft_mode=0 --batch_size=8

python code/train.py --model_name=VITs16_v3-1 --DA_name=DA0 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v3-2 --DA_name=DA0 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v3-3 --DA_name=DA0 --ft_mode=0 --batch_size=8

python code/train.py --model_name=VITs16_baseline --DA_name=DA4 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v0 --DA_name=DA4 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v1 --DA_name=DA4 --ft_mode=0 --batch_size=8

python code/train.py --model_name=VITs16_v2-1 --DA_name=DA4 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v2-2 --DA_name=DA4 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v2-3 --DA_name=DA4 --ft_mode=0 --batch_size=8

python code/train.py --model_name=VITs16_v3-1 --DA_name=DA4 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v3-2 --DA_name=DA4 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v3-3 --DA_name=DA4 --ft_mode=0 --batch_size=8

python code/train.py --model_name=VITs16_baseline --DA_name=DA5 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v0 --DA_name=DA5 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v1 --DA_name=DA5 --ft_mode=0 --batch_size=8

python code/train.py --model_name=VITs16_v2-1 --DA_name=DA5 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v2-2 --DA_name=DA5 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v2-3 --DA_name=DA5 --ft_mode=0 --batch_size=8

python code/train.py --model_name=VITs16_v2-1 --DA_name=DA5 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v2-2 --DA_name=DA5 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v2-3 --DA_name=DA5 --ft_mode=0 --batch_size=8

python code/train.py --model_name=VITs16_v3-1 --DA_name=DA5 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v3-2 --DA_name=DA5 --ft_mode=0 --batch_size=8
python code/train.py --model_name=VITs16_v3-3 --DA_name=DA5 --ft_mode=0 --batch_size=8