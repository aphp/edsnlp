#!/bin/bash 
#SBATCH --job-name=ner_med_training
#SBATCH -t 24:00:00 
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition gpuV100
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable 
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
# your code here :

echo starting
conda activate pierrenv
cd '/export/home/cse200093/Jacques_Bio/BioMedics/eds-medic'

echo ---- Building dvc.yaml ----

python -m spacy project dvc

echo ---- Saving Brat files ----

python -m spacy project run save_to_brat --force

echo ---------------
