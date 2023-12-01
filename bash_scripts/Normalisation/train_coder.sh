source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh

cd ~/scratch
source BioMedics/.venv/bin/activate
conda deactivate

eds-toolbox slurm submit --config BioMedics/normalisation/training/train_coder_slurm.cfg -c "python BioMedics/normalisation/training/train.py --umls_dir BioMedics/data/umls/2021AB/ --model_name_or_path word-embedding/coder_eds  --output_dir word-embedding/coder_eds --gradient_accumulation_steps 8 --train_batch_size 1024 --lang eng_fr"
