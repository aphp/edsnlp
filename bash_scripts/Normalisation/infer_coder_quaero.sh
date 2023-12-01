#!/bin/bash 
#SBATCH --job-name=ner_med_training
#SBATCH -t 48:00:00 
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition gpuV100
#SBATCH --output=log_infer_coder/slurm-%j-stdout.log
#SBATCH --error=log_infer_coder/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script

cd 'data/scratch/cse200093/BioMedics'
source .venv/bin/activate
conda deactivate


echo -----------------
echo NORMALIZE ANNOTATED DOCS
echo -----------------

python normalisation/inference/main.py normalisation/data/CRH/annotated_umls_snomed_full.json normalisation/data/pred_coder_eds/annotated_bio_micro.json

echo -----------------
echo NORMALIZE QUAERO DOCS
echo -----------------

python normalisation/inference/main.py normalisation/data/quaero_bio_micro.json normalisation/data/pred_coder_eds/quaero_bio_micro.json

echo --NORMALIZATION_FINISHED---

echo ---------------
