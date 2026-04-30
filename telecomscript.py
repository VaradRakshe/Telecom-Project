from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import broadcast

# ----------------------------
# Spark Session
# ----------------------------
spark = SparkSession.builder \
    .appName("TELECOM") \
    .getOrCreate()

# ----------------------------
# Spark Config
# ----------------------------
spark.conf.set("spark.sql.shuffle.partitions", 300)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# ----------------------------
# Load Excel
# ----------------------------
df = spark.read.format("com.crealytics.spark.excel") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(r"C:\Users\91838\Desktop\telecom project\telecom data for pyspark.xlsx")

df.show(5)

# ----------------------------
# Column Standardization (CRITICAL FIRST STEP)
# ----------------------------
df = df.toDF(*[
    c.strip()
     .replace(" ", "_")
     .replace("%", "pct")
     .replace("-", "_")
     .replace("[", "")
     .replace("]", "")
    for c in df.columns
])

# Debug (always keep during development)
print("Columns after cleaning:")
print(df.columns)

# ----------------------------
# Repartition (AFTER cleaning)
# ----------------------------
df = df.repartition("Date", "Cell_ID")
df.show(6)
# ----------------------------
# Casting
# ----------------------------
numeric_cols = [
    "Cell_Availabililty",
    "Session_Setup_Success_Rate",
    "VoLTE_Drop_Rate_pct",
    "Handover_Success_Rate_pct",
    "Traffic_24Hrs_GB",
    "DL_PRB_utilizationpct",
    "CQI",
    "IP_Throughput_Mbps",
    "RRC_Connected_Users",
    "Peak_RRC_Connected",
    "Average_TA",
    "Mute_Call_Ratepct"
]

for col in numeric_cols:
    df = df.withColumn(col, F.col(col).cast("double"))

# Fix Date format if needed
df = df.withColumn("Date", F.to_date("Date"))
df.show(2)

# ----------------------------
# Data Cleaning
# ----------------------------
df = df.dropna(subset=["Cell_ID", "Date"]).fillna(0)

# Cache (and trigger execution)
df.cache()
df.count()

# ----------------------------
# Rolling Window KPIs
# ----------------------------
window_spec = Window.partitionBy("Cell_ID").orderBy("Date").rowsBetween(-6, 0)

df = df.withColumn("rolling_cssr", F.avg("Session_Setup_Success_Rate").over(window_spec)) \
       .withColumn("rolling_drop", F.avg("VoLTE_Drop_Rate_pct").over(window_spec)) \
       .withColumn("rolling_tp", F.avg("IP_Throughput_Mbps").over(window_spec))

# ----------------------------
# Z-score (Anomaly Detection)
# ----------------------------
stats = df.groupBy("Cell_ID").agg(
    F.mean("IP_Throughput_Mbps").alias("mean_tp"),
    F.stddev("IP_Throughput_Mbps").alias("std_tp")
)

df = df.join(broadcast(stats), "Cell_ID")

df = df.withColumn(
    "zscore_tp",
    (F.col("IP_Throughput_Mbps") - F.col("mean_tp")) / F.col("std_tp")
).withColumn(
    "tp_anomaly",
    F.when(F.abs(F.col("zscore_tp")) > 2, 1).otherwise(0)
)

# ----------------------------
# Min-Max Scaling
# ----------------------------
minmax_df = df.select(
    *[F.min(c).alias(f"{c}_min") for c in numeric_cols],
    *[F.max(c).alias(f"{c}_max") for c in numeric_cols]
)

minmax_df = broadcast(minmax_df)
df = df.crossJoin(minmax_df)

def normalize(col):
    return (
        (F.col(col) - F.col(f"{col}_min")) /
        (F.col(f"{col}_max") - F.col(f"{col}_min") + F.lit(1e-6))
    )

df = df.withColumn("norm_cssr", normalize("Session_Setup_Success_Rate")) \
       .withColumn("norm_drop", 1 - normalize("VoLTE_Drop_Rate_pct")) \
       .withColumn("norm_ho", normalize("Handover_Success_Rate_pct")) \
       .withColumn("norm_prb", 1 - normalize("DL_PRB_utilizationpct")) \
       .withColumn("norm_cqi", normalize("CQI")) \
       .withColumn("norm_tp", normalize("IP_Throughput_Mbps"))

# ----------------------------
# Health Score Calculation
# ----------------------------
df = df.withColumn(
    "health_score",
    (
        F.col("norm_cssr") * 0.2 +
        F.col("norm_drop") * 0.2 +
        F.col("norm_ho") * 0.15 +
        F.col("norm_prb") * 0.1 +
        F.col("norm_cqi") * 0.15 +
        F.col("norm_tp") * 0.2
    )
)
df.show(3)
# ----------------------------
# Ranking Worst Cells
# ----------------------------
rank_window = Window.partitionBy("Date").orderBy(F.col("health_score").asc())

df_ranked = df.withColumn("rank", F.row_number().over(rank_window))
df_ranked.show(4)
worst_cells = df_ranked.filter(F.col("rank") <= 10)
worst_cells.show(10)
# ----------------------------
# Write Output
# ----------------------------