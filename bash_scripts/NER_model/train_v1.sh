#!/bin/bash
#SBATCH --job-name=ner_med_training
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition gpuV100
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script

cd 'data/scratch/cse200093/BioMedics/NER_model'
source ../.venv/bin/activate
conda deactivate

echo -----------------
echo CONVERT DOCS
echo -----------------

python scripts/convert.py --lang eds --input-path ./data/NLP_diabeto/train_test --output-path ./corpus/train_test.spacy
python scripts/convert.py --lang eds --input-path ./data/NLP_diabeto/val --output-path ./corpus/dev.spacy

echo -----------------
echo TRAIN ON DOCS
echo -----------------

python -m spacy train ./configs/config_v1.cfg --output ./training/model_v1/ --paths.train ./corpus/train_test.spacy --paths.dev ./corpus/dev.spacy --nlp.lang eds --gpu-id 0

echo -----------------
echo INFER LUPUS DOCS
echo -----------------

python ./scripts/infer.py --model ./training/model_v1/model-best/ --input ../data/CRH/raw/lupus_erythemateux_dissemine/ --output ../data/CRH/pred/lupus_erythemateux_dissemine/ --format brat

echo -----------------
echo INFER MALADIE TAKAYASU DOCS
echo -----------------

python ./scripts/infer.py --model ./training/model_v1/model-best/ --input ../data/CRH/raw/maladie_de_takayasu/ --output ../data/CRH/pred/maladie_de_takayasu/ --format brat

echo -----------------
echo INFER SCLERODERMIE SYSTEMIQUE DOCS
echo -----------------

python ./scripts/infer.py --model ./training/model_v1/model-best/ --input ../data/CRH/raw/sclerodermie_systemique/ --output ../data/CRH/pred/sclerodermie_systemique/ --format brat

echo -----------------
echo INFER SAPL DOCS
echo -----------------

python ./scripts/infer.py --model ./training/model_v1/model-best/ --input ../data/CRH/raw/syndrome_des_anti-phospholipides/ --output ../data/CRH/pred/syndrome_des_anti-phospholipides/ --format brat


echo --Inference_done---

echo ---------------
