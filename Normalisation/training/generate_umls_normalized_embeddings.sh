#!/bin/bash
#SBATCH --job-name=generate_umls_normalized_embeddings.sh
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:t4:1
#SBATCH -N1-1
#SBATCH -c2
#SBATCH --mem=40000
#SBATCH -p gpuT4
#SBATCH -w bbs-edsg28-p012
#SBATCH --output=./log/%x-%j.out
#SBATCH --error=./log/%x-%j.err
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable

source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
# your code here :
echo starting
conda activate pierrenv
#cd /export/home/cse200093/Jacques_Bio/normalisation/py_files
which python
pwd
python /export/home/cse200093/Jacques_Bio/normalisation/py_files/generate_umls_normalized_embeddings.py
echo end
#export PYTHONPATH=/export/home/cse200093/nlstruct-master/:.
#which python
#python ./main.py
