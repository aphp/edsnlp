#!/bin/bash 
#SBATCH --job-name=ner_med_training
#SBATCH -t 24:00:00 
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition gpuT4
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable 
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
# your code here :

echo starting
conda activate pierrenv

cd '/export/home/cse200093/Jacques_Bio/BioMedics/eds-medic'

#python ./scripts/infer.py --model ~/RV_Inter_conf/model-best/ --input ~/RV_Inter_conf/unnested_sosydiso_qualifiers_final/test_ --output ~/RV_Inter_conf/usqf_pred/ --format brat

python ./scripts/infer.py --model ./training/model-best/ --input ../data/lupus_erythemateux_dissemine_raw --output ../data/lupus_erythemateux_dissemine_pred/ --format brat