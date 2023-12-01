source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh

cd ~/scratch/BioMedics
source .venv/bin/activate
conda deactivate


echo -----------------
echo EXTRACTING MEASUREMENT VALUES AND UNITS USING BIO_COMP LABEL AND RULES.
echo -----------------

echo -----------------
echo EXTRACT MEASUREMENT FROM MALADIE TAKAYASU
echo -----------------

python extract_measurement/main.py ./data/CRH/pred/maladie_de_takayasu ./data/bio_results/maladie_de_takayasu

echo -----------------
echo EXTRACT MEASUREMENT FROM LUPUS
echo -----------------

python extract_measurement/main.py ./data/CRH/pred/lupus_erythemateux_dissemine ./data/bio_results/lupus_erythemateux_dissemine

echo -----------------
echo EXTRACT MEASUREMENT FROM SCLERODERMIE SYSTEMIQUE
echo -----------------

python extract_measurement/main.py ./data/CRH/pred/sclerodermie_systemique ./data/bio_results/sclerodermie_systemique

echo -----------------
echo EXTRACT MEASUREMENT FROM SAPL
echo -----------------

python extract_measurement/main.py ./data/CRH/pred/syndrome_des_anti-phospholipides ./data/bio_results/syndrome_des_anti-phospholipides


echo --EXTRACTION_FINISHED---

echo ---------------
