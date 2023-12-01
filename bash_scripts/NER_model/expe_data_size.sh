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

for i in 5 10 15 20 25 30 35 40 45 50 55 60 62
do
    echo -----------------
    echo CONVERT $i DOCS
    echo -----------------

    python scripts/convert.py --lang eds --input-path ./data/NLP_diabeto/train --output-path ./corpus/expe_data_size/train_$i.spacy --n-limit $i

    echo -----------------
    echo TRAIN ON $i DOCS
    echo -----------------

    python -m spacy train configs/config.cfg --output ./training/expe_data_size/model_$i/ --paths.train ./corpus/expe_data_size/train_$i.spacy --paths.dev ./corpus/dev.spacy --nlp.lang eds --gpu-id 0


    echo -----------------
    echo REMOVE MODEL LAST
    echo -----------------

    rm -rf ./training/expe_data_size/model_$i/model-last

    echo -----------------
    echo INFER TEST DOCS WITH MODEL TRAINED ON $i DOCS
    echo -----------------

    python ./scripts/infer.py --model ./training/expe_data_size/model_$i/model-best/ --input ./data/NLP_diabeto/test/ --output ./data/NLP_diabeto/expe_data_size/pred_$i/ --format brat

    echo -----------------
    echo EVALUATE MODEL TRAINED ON $i DOCS
    echo -----------------

    python ./scripts/evaluate.py ./training/expe_data_size/model_$i/model-best ./corpus/test.spacy --output ./training/expe_data_size/model_$i/test_metrics.json --docbin ./data/NLP_diabeto/expe_data_size/pred_$i.spacy --gpu-id 0

done


echo --Training_done---

echo ---------------
