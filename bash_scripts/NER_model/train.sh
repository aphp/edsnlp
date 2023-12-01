#!/bin/bash
#SBATCH --job-name=ner_med_training
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition gpuV100
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script

conda activate pierrenv
cd '/export/home/cse200093/Jacques_Bio/BioMedics/eds-medic'


python -m spacy project dvc

echo dvc.yml built succesfully

echo -----------------
echo CONVERT
echo -----------------

python -m spacy project run convert

echo -----------------
echo TRAIN
echo -----------------

dvc repro -f 2>&1 | tee training/train.log

echo --Training_done---

echo ---------------
