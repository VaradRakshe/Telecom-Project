from pyspark.sql import SparkSession
spark_excel = SparkSession.builder.appName("Telecom").config("spark.jars.packages", "com.crealytics:spark-excel_2.13:3.5.1_0.20.4").getOrCreate()

df = spark_excel.read.format("com.crealytics.spark.excel")\
.option("header", "true")\
.option("inferSchema", "true")\
.load(r"C:\Users\91838\Desktop\telecom project\telecom data for pyspark.xlsx")

#print(df.show(5))

# -------- 3. Data Cleaning ---------

from pyspark.sql.functions import col, to_date
df_clean = df.dropna(subset=["Cell ID", "SITE ID"])

df_clean = df_clean.filter(
    (col("Cell Availabililty") >= 0) &
    (col("Session Setup Success Rate") <= 100)
)

df_clean = df_clean.withColumn("Date", to_date ("Date", "dd-MMM-yy"))

# -------- 4. Site level Aggregation ---------

from pyspark.sql.functions import avg, sum, max

site_daily = df_clean.groupBy("SITE ID","Date").agg(
avg("Cell Availabililty").alias("Avg_Cell_Availability"),
avg("Session Setup Success Rate").alias("Avg_Session_SR"),
avg("VoLTE_Drop_Rate_%").alias("Avg_VoLTE_Drop"),
avg("Handover_Success_Rate_%").alias("Avg_HO_SR"),
sum("Traffic-24Hrs [GB]").alias("Total_Traffic_GB"),
avg("DL_PRB_utilization%").alias("Avg_PRB_Utilization"),
avg("CQI").alias("Avg_CQI"),
avg("IP_Throughput_Mbps").alias("Avg_Throughput"),
max("Peak RRC Connected").alias("Peak_RRC")
)

site_daily.show()

# -------- 5. Detecting Network Congestion ---------

congestion_cells = df_clean.filter(
    (col("DL_PRB_utilization%") > 85) &
    (col("IP_Throughput_Mbps") < 5)
    )

congestion_cells.select(
    "Cell ID","SITE ID","DL_PRB_utilization%","IP_Throughput_Mbps"
    ).show()


# --------  6. VoLTE Quality Issues ---------

volte_issue_cells = df_clean.filter(
    (col("VoLTE_Drop_Rate_%") > 2) |
    (col("Mute_Call Rate%") > 1)
    )

volte_issue_cells.show()

# --------  7. Coverage Issue Detection ---------

coverage_issue = df_clean.filter(
    (col("Average_TA") > 600) &
    (col("CQI") < 7)
    )

coverage_issue.show()

# --------  8. Window Function for Traffic Trend ---------

from pyspark.sql.window import Window
from pyspark.sql.functions import lag

windowSpec = Window.partitionBy("Cell ID").orderBy("Date")

traffic_trend = df_clean.withColumn(
    "Prev_Traffic",
    lag("Traffic-24Hrs [GB]").over(windowSpec)
    )

traffic_trend.show()

# 9. Save Output
site_daily.write .mode("overwrite") .parquet("/analytics/telecom/site_kpi_summary")
