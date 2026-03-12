"""
PySpark preprocessing pipeline for ClinSense (Databricks native).
Replicates the Pandas filtering logic from src/data/loader.py and 
scripts/optimal_filter_and_train.py using a scalable Databricks architecture.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, coalesce, lit, length, size, split, count, avg

def main():
    # 1. Initialize SparkSession (Databricks manages the session, but getOrCreate is standard)
    spark = SparkSession.builder.getOrCreate()
    
    # 2. Use dbutils.widgets for configurable input/output paths
    # In Databricks notebooks, dbutils is automatically available
    try:
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
    except ImportError:
        try:
            import IPython
            dbutils = IPython.get_ipython().user_ns.get("dbutils")
        except Exception:
            dbutils = None

    if dbutils:
        dbutils.widgets.text("input_path", "gs://clinsense-prod-123456/data/raw/mtsamples.csv", "Input CSV Path")
        dbutils.widgets.text("output_path", "gs://clinsense-prod-123456/data/processed/mtsamples_filtered.parquet", "Output Parquet Path")
        input_path = dbutils.widgets.get("input_path")
        output_path = dbutils.widgets.get("output_path")
    else:
        # Fallback values if running outside of a Databricks notebook context
        input_path = "gs://clinsense-prod-123456/data/raw/mtsamples.csv"
        output_path = "gs://clinsense-prod-123456/data/processed/mtsamples_filtered.parquet"

    print(f"Reading CSV from: {input_path}")
    
    # 3. Read CSV from Databricks FileStore or DBFS path
    # multiLine=True and escape='"' handle text that contains newlines
    df = spark.read.csv(input_path, header=True, inferSchema=True, multiLine=True, escape='"')

    # Handle various naming conventions from original loader.py
    if "Unnamed: 0" in df.columns:
        df = df.withColumnRenamed("Unnamed: 0", "note_id")
    elif "0" in df.columns:
        df = df.withColumnRenamed("0", "note_id")
    if "specialty" in df.columns and "medical_specialty" not in df.columns:
        df = df.withColumnRenamed("specialty", "medical_specialty")
    if "text" in df.columns and "transcription" not in df.columns:
        df = df.withColumnRenamed("text", "transcription")

    # 4. Fill NaN + strip whitespace on transcription and medical_specialty columns
    df = df.withColumn("transcription", trim(coalesce(col("transcription").cast("string"), lit(""))))
    df = df.withColumn("medical_specialty", trim(coalesce(col("medical_specialty").cast("string"), lit(""))))

    # 5. Cache the DataFrame after initial load to optimize subsequent multiple actions (counts, stats)
    df.cache()
    
    initial_count = df.count()
    print(f"Initial row count: {initial_count}")

    # 6. Drop rows with transcription length < 50 characters
    df = df.filter(length(col("transcription")) >= 50)
    
    count_after_length = df.count()
    print(f"Row count after removing transcription length < 50: {count_after_length}")

    # Calculate word count for each transcription dynamically using regex split on whitespace
    df = df.withColumn("word_count", size(split(col("transcription"), r"\s+")))

    # 7. Single-pass aggregation to compute samples per specialty and average word count
    specialty_stats = df.groupBy("medical_specialty").agg(
        count("*").alias("sample_count"),
        avg("word_count").alias("avg_word_count")
    )
    
    # 8. Filter specialties:
    #   - Minimal 150 samples
    #   - Minimum average word count of 100
    valid_specialties = specialty_stats.filter(
        (col("sample_count") >= 150) & 
        (col("avg_word_count") >= 100)
    )
    
    # 9. Hardcoded specialty list to match already-deployed model (sync with Option 1 fix)
    top_8_specialties = [
        "Cardiovascular / Pulmonary",
        "Gastroenterology",
        "Neurology",
        "Obstetrics / Gynecology",
        "Orthopedic",
        "Radiology",
        "SOAP / Chart / Progress Notes",
        "Urology"
    ]
    print(f"Using hardcoded model specialties: {top_8_specialties}")

    # 10. Filter the main DataFrame to keep only the top 8 specialties
    # This is a highly efficient single-pass filter natively broadcasted
    final_df = df.filter(col("medical_specialty").isin(top_8_specialties))
    
    # Drop the temporary word_count column if we just want to store the raw data features
    final_df = final_df.drop("word_count")
    
    final_count = final_df.count()
    print(f"Final row count: {final_count}")

    # 11. Write output as Parquet to DBFS
    print(f"Writing Parquet output to: {output_path}")
    final_df.write.mode("overwrite").parquet(output_path)
    
    print("Preprocessing pipeline completed.")

if __name__ == "__main__":
    main()
