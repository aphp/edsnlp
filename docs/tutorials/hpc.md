# Running an existing model on HPC (e.g. Slurm)

This tutorial shows how to run an existing deep-learning based EDS-NLP model (for example the
public pseudonymisation model [eds-pseudo-public](https://eds-pseudo-public.streamlit.app) efficiently
on a cluster. In a Clinical Data Warehouse like AP-HP's, most research projects might want to:

1. first fetch a corpus of documents with PySpark. Depending on your computing setup, this might run on a specific cluster like Hadoop/YARN.
2. run the NLP model on these notes. This is often best done on a GPU cluster, for instance one managed by Slurm.

## Python inference script

Let's start by the Python NLP inference script. We’ll write an inference script that:

- loads an existing model, e.g. `AP-HP/dummy-ner` which annotates entities of the [DEFT 2020 dataset](https://hal.science/hal-03095262/document) on documents.
- reads notes from a Parquet dataset (e.g. exported from Spark)
- applies the model on these notes
- writes entities back to a new Parquet dataset (e.g. to be re-imported in Spark)

```python { title="inference.py" }
import logging
import os
from datetime import datetime
from typing import Union

import confit
import pyarrow.fs

import edsnlp


def make_fs(path: str, endpoint: str = None):
    """
    This function can be used to define the filesystem explicitly.
    Otherwise, it's automatically created from the path
    (ex: "s3://", "hdfs://", ...) using default parameters.
    """

    # For instance, if you have a s3 volume (S3 is not necessarily AWS !)
    # you can use the S3 filesystem and provide credentials as env vars.
    if path.startswith("s3://"):
        return pyarrow.fs.S3FileSystem(
            access_key=os.getenv("S3_ACCESS_KEY"),
            secret_key=os.getenv("S3_SECRET_KEY"),
            endpoint_override=os.getenv("S3_ENDPOINT"),
        )
    return None


app = confit.Cli()  #(1)!


@app.command("inference")
def main(
    *,
    input_path: str,
    output_path: str,
    model_name: str = "AP-HP/dummy-ner",
    batch_size: str = "32 docs",
    show_progress: bool = False,
    output_file_size: Union[int, str] = 10_000,
):
    """
    Run inference on a corpus of notes stored in Parquet format.

    Parameters
    ----------
    input_path : str
        Input Parquet path (e.g. s3://bucket/notes/ or hdfs path)
    output_path : str
        Output Parquet path (e.g. s3://bucket/note_nlp/ or hdfs path)
    model_name : str
        Model to load: local path, installed model package or EDS-NLP
        compatible Hub repo (e.g. 'AP-HP/eds-pseudo-public')
    batch_size : str
        Batch size expression (e.g. '32 docs', '8000 words')
    show_progress : bool
        Show progress bars
    output_file_size : Union[int, str]
        Size per Parquet file (e.g. '1000 docs', '40000 words')
        in the output dataset
    """

    logging.info("Model loading started")
    nlp = edsnlp.load(model_name)
    # Do anything to the model here
    print(nlp)
    logging.info("Model loading done")

    input_fs = make_fs(input_path)
    output_fs = make_fs(output_path)

    print(f"Job started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Read OMOP-like parquet (note_id, person_id, note_text, ...)
    docs = edsnlp.data.read_parquet(
        path=input_path,
        converter="omop",
        filesystem=input_fs,
        read_in_worker=True,
        doc_attributes=["note_id", "person_id"],  #(2)!
    )

    # Apply the model lazily
    docs = docs.map_pipeline(nlp)

    # Configure multiprocessing with automatic resource detection
    docs = docs.set_processing(
        backend="multiprocessing",
        batch_size=batch_size,
        show_progress=show_progress,
        # You can set num_cpu_workers and num_gpu_workers here,
        # otherwise they are auto-detected
    )

    # Write entities to parquet, with a fallback when no entity
    # Feel free to change the output format here
    def doc_to_rows(doc):
        rows = [
            dict(
                note_id=getattr(doc._, "note_id", None),
                person_id=getattr(doc._, "person_id", None),
                offset_begin=ent.start_char,
                offset_end=ent.end_char,
                label=ent.label_,
                snippet=ent.text,
                date=getattr(ent._, 'date'),
                # You can add other ent attributes here
                # like ent._.certainty, ent._.family, etc.
                nlp_system=model_name,
            )
            for ent in doc.ents
        ]
        return rows or [
            dict(
                note_id=getattr(doc._, "note_id", None),
                person_id=getattr(doc._, "person_id", None),
                offset_begin=0,
                offset_end=0,
                label="EMPTY",
                snippet="",
                date=None,
                # You can add other ent attributes here
                nlp_system=model_name,
            )
        ]

    # We declare here where we want to write the output
    # All writers trigger the execution by default (unless execute=False)
    docs.write_parquet(
        path=output_path,
        overwrite=True,
        batch_size=output_file_size,
        converter=doc_to_rows,
        filesystem=output_fs,
    )

    print(f"Job done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    app()
```

1. We use [confit](https://github.com/aphp/confit) to create a CLI application and enforce parameter types.

!!! tip "Converters and schemas"

    - If your input is not OMOP-like (ie with `note_text` and `note_id` columns),
      provide your own reader converter instead of `converter="omop"` (see the [Converters](/data/converters) page).
    - See the tutorial [Processing multiple texts](/tutorials/multiple-texts) for more about batching expressions (`batch_size`) and the `backend` options.

## Accessing computation resources on Slurm

Slurm is a workload manager for HPC clusters. You request resources (CPUs, memory, GPUs, time) and submit jobs with scripts. Key
Below is a Slurm script that activates your environment, shows GPU info, and runs the inference script.

```sbatch { title="job.sh" }
# Name your job clearly to find it in queues and reports.
#SBATCH --job-name=nlp
# Walltime limit. Increase if you hit the time limit.
#SBATCH --time 1:00:00
# For instance here, we request one V100 GPU. Adapt to what your cluster
# provides, and contact your admin if unsure.
#SBATCH --gres=gpu:v100:1
#SBATCH --partition gpuV100
# Single-node job with 4 CPU cores for rule-based pipes, preprocessing,
# collation, postprocessing.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
# RAM (!= GPU VRAM !) per node (in MB), adjust if you hit OOM errors.
#SBATCH --mem=50000
# Container config (if your Slurm allows this). Adapt to your cluster
# setup and contact your admin if unsure.
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh --container-mounts=/export/home/$USER:/export/home/$USER,/export/home/share:/export/home/share,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable --container-workdir=/
# Stdout/stderr file patterns with `%j` expanded to the job ID.
# You can put these in a logs/ directory if you prefer, but MAKE SURE
# THAT THIS DIRECTORY EXISTS BEFORE SUBMITTING !
#SBATCH --output=slurm-%j-stdout.log
#SBATCH --error=slurm-%j-stderr.log

set -euo pipefail
# Setup the env. Simple setup for AP-HP cluster below
# Refer to your HPC documentation for your own setup.
/etc/start.sh
export HADOOP_HOME=/usr/local/hadoop
export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`
export ARROW_LIBHDFS_DIR=/usr/local/hadoop/usr/lib/
source "$HOME/.user_conda/miniconda/etc/profile.d/conda.sh"

# Activate your environment(s), e.g. conda/venv/uv or a mix of these
conda activate your-conda-env
source path/to/your/project/.venv/bin/activate

# You can install packages here. Doing this here can be useful to
# ensure installed versions match the deployment env, for instance
# glibc, CUDA versions, etc. Otherwise, install in your env beforehand.
pip install "edsnlp[ml]" "pyarrow<17"

# Check available GPUs
nvidia-smi

cd path/to/your/project
python inference.py \
  --model_name "AP-HP/dummy-ner" \
  --input_path "hdfs:///user/USERNAME/notes/" \
  --output_path "hdfs:///user/USERNAME/nlp_results/" \
  --batch_size "10000 words" \
  --output_file_size "1000 docs" \
  --show_progress
```

## Run and monitor the job

1. Launch the job and store the job id in a JOB_ID variable:
    ```bash { data-md-color-scheme="slate" }
    JOB_ID=$(sbatch job.sh | awk '{print $4}') && echo "Job: $JOB_ID"
    ```
    ```
    Job ID: 123456
    ```

2. See the current running and pending jobs.with `squeue`
    ```bash { data-md-color-scheme="slate" }
    squeue
    ```
    ```
    JOBID PARTITION NAME     USER ST  TIME NODES NODELIST(REASON)
    123456  gpuV100  nlp USERNAME R   0:10     1      gpu-node-01
    ```
- Cancel the job if needed with:
    ```bash { data-md-color-scheme="slate" }
    scancel $JOB_ID
    ```

- Follow the logs in real time with. See the above #SBATCH directive comment to put them in a directory if needed.
  ```bash { data-md-color-scheme="slate" }
  tail -f -n+0 slurm-$JOB_ID-std*.log
  ```

## Fetching data with PySpark

The above job requires a Parquet dataset as input. You can use PySpark to extract notes from your CDW and write them to Parquet.
In theory, you could run end-to-end with Spark using
```python { .no-check }
docs = edsnlp.data.from_spark(...)
```
However, this interleaves Spark’s distributed CPU scheduling with the GPU-based inference, often mobilizing many CPUs in an uncoordinated way while documents stream through both PySpark and the GPU workers.

A more robust pattern is to decouple document selection from inference. In a Spark-enabled notebook or a Spark-submit job:

1. Extract your input corpus with Spark, write to Parquet (HDFS or S3)
   ```python { .no-check }
   from pyspark.sql import SparkSession

   spark = SparkSession.builder.getOrCreate()

   note = spark.sql("""
       SELECT note_id, person_id, note_text
       FROM your_database.note
       WHERE note_datetime >= '2024-01-01' and note_text IS NOT NULL
       LIMIT 10000
   """)

   note.write.mode("overwrite").parquet("hdfs:///user/USERNAME/notes/")
   ```

2. Run the Slurm GPU inference on that Parquet dataset, as in sections above (point your `--input_path` to the Parquet location and `--output_path` to a destination Parquet directory).

3. Load the inference results back into Spark if needed (aggregation, joins, etc.)
   ```python { .no-check }
   note_nlp = spark.read.parquet("hdfs:///user/USERNAME/nlp_results/")
   note_nlp.createOrReplaceTempView("note_nlp")

   # Example: count entities per label
   spark.sql("""
       SELECT label, COUNT(*) AS n
       FROM note_nlp
       GROUP BY label
       ORDER BY n DESC
   """).show()
   ```

This approach keeps GPU inference scheduling independent of Spark, avoids excessive CPU pressure, and is easier to monitor and reason about.
