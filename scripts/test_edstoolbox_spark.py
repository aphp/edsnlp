from pyspark.sql import SparkSession
from edstoolbox import SparkApp  # type: ignore

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sql = spark.sql


from pyspark.sql import functions as F

def patient_selection():
    sql("use edsomop_dev")
    note = sql("select * from note")
    note = note.select(["note_id", "note_class_source_value", "note_text"])

    note = note.limit(200).repartition(8)
    from edsnlp.processing.distributed import custom_pipe, pyspark_type_finder

    from edsnlp.processing import pipe
    import spacy
    from edsnlp.matchers.utils import get_text

    config = dict(
        lowercase=False,
        accents=True,
        quotes=False,
        pollution=True,
    )

    nlp = spacy.blank("fr")
    nlp.add_pipe("eds.normalizer", config=config)

    int_type = pyspark_type_finder(1)
    str_type = pyspark_type_finder("")

    def results_extractor(doc):
        spans = doc.spans["pollutions"]
        results = []
        note_id = doc._.note_id
        for span in spans:
            results.append(
                dict(
                    note_id=note_id,
                    text=span.text,
                    start_char=span.start_char,
                    end_char=span.end_char,
                )
            )

        return results


    dtypes = dict(note_id=str_type, text=str_type, start_char=int_type, end_char=int_type)

    note_nlp = pipe(
        note=note,
        nlp=nlp,
        results_extractor=results_extractor,
        dtypes=dtypes,
    )

    note_nlp.show()


# Initialize app
app = SparkApp("patient_selection")

@app.submit
def run(spark, sql, config):
    patient_selection()


if __name__ == "__main__":

    app.run()