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
echo NORMALIZE LUPUS DOCS
echo -----------------

python normalisation/inference/main.py data/bio_results/lupus_erythemateux_dissemine/pred_with_extraction.json data/bio_results/lupus_erythemateux_dissemine/norm_coder_all.json

echo -----------------
echo NORMALIZE MALADIE TAKAYASU DOCS
echo -----------------

python normalisation/inference/main.py data/bio_results/maladie_de_takayasu/pred_with_extraction.json data/bio_results/maladie_de_takayasu/norm_coder_all.json

echo -----------------
echo NORMALIZE SCLERODERMIE SYSTEMIQUE DOCS
echo -----------------

python normalisation/inference/main.py data/bio_results/sclerodermie_systemique/pred_with_extraction.json data/bio_results/sclerodermie_systemique/norm_coder_all.json

echo -----------------
echo NORMALIZE SAPL DOCS
echo -----------------

python normalisation/inference/main.py data/bio_results/syndrome_des_anti-phospholipides/pred_with_extraction.json data/bio_results/syndrome_des_anti-phospholipides/norm_coder_all.json

echo --NORMALIZATION_FINISHED---

echo ---------------
