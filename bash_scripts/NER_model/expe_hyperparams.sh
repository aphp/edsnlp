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

cd 'data/scratch/cse200093/BioMedics/NER_model'
source ../.venv/bin/activate
conda deactivate

echo -----------------
echo CONVERT DOCS
echo -----------------

python scripts/convert.py --lang eds --input-path ./data/NLP_diabeto/train --output-path ./corpus/train.spacy
python scripts/convert.py --lang eds --input-path ./data/NLP_diabeto/test --output-path ./corpus/test.spacy
python scripts/convert.py --lang eds --input-path ./data/NLP_diabeto/val --output-path ./corpus/dev.spacy

echo -----------------
echo TRAIN ON DOCS
echo -----------------

python -m spacy train ./configs/config.cfg --output ./training/expe_section/model_all_labels/ --paths.train ./corpus/train.spacy --paths.dev ./corpus/dev.spacy --nlp.lang eds --gpu-id 0

echo -----------------
echo INFER TEST DOCS
echo -----------------

python ./scripts/infer.py --model ./training/expe_section/model_all_labels/model-best --input ./data/NLP_diabeto/test/ --output ./data/NLP_diabeto/expe_section/pred_model_all_labels/ --format brat


echo -----------------
echo EVALUATE MODEL
echo -----------------

python ./scripts/evaluate.py ./training/expe_section/model_all_labels/model-best ./corpus/test.spacy --output ./training/expe_section/model_all_labels/test_metrics.json --docbin ./data/NLP_diabeto/expe_section/pred_model_all_labels.spacy --gpu-id 0

echo --Training_done---

echo ---------------
