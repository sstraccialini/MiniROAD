#!/bin/bash
#SBATCH --account=3199302
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-tsu_clip-%j.out
#SBATCH --cpus-per-task=8

# ====== USER CONFIG ======
MYID=3199302
BASE_HOME=/mnt/beegfsstudents/home/$MYID
USER_HOME=/home/$MYID

# ====== ENV SETUP ======
source ~/.bashrc
conda activate mstemba

cd $BASE_HOME/MiniROAD

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ====== RUN CODE ======
# TSU with I3D features
python main.py --config configs/tsu_i3d.yaml --feature_root ../ASO-Temba/data/tsu_features_i3d

# TSU with CLIP features
python main.py --config configs/tsu_clip.yaml --feature_root ../ASO-Temba/data/tsu_features_clip_l14

# Charades with I3D features
python main.py --config configs/charades_i3d.yaml --feature_root ../ASO-Temba/data/charades_features_i3d

# Charades with CLIP features
python main.py --config configs/charades_clip.yaml --feature_root ../ASO-Temba/data/charades_features_clip