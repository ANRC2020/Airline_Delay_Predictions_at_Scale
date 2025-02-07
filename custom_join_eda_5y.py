# Databricks notebook source
# General
import re
import ast
import time

# Data
import numpy as np
import pandas as pd

# Spark Pandas API
import pyspark.pandas as ps

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

# Spark SQL API
from pyspark.sql import Window, DataFrame
from pyspark.sql import functions as F

# Specific Spark SQL functions
from pyspark.sql.functions import row_number, col, when, sum, row_number, concat, to_timestamp
from pyspark.sql.functions import col, isnan, when, count, col, split, trim, lit, avg, lpad, floor
from pyspark.sql.functions import expr, mean, stddev, randn, to_date, round, upper, trim, countDistinct

# Spark ML Lib functions
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors

# Feature pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler

# Regression model
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# Mount blob storae and grant team access
blob_container  = "261project"       # The name of your container created in https://portal.azure.com
storage_account = "261teamderm"  # The name of your Storage account created in https://portal.azure.com
secret_scope = "261teamderm"           # The name of the scope created in your local computer using the Databricks CLI
secret_key = "261key"             # The name of the secret key created in your local computer using the Databricks CLI
team_blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket
mids261_mount_path      = "/mnt/mids-w261" # the 261 course blob storage is mounted here.
# SAS Token: Grant the team limited access to Azure Storage resources
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
  )

# COMMAND ----------

#fpath_3m = f'{team_blob_url}/corrected_custom_join_1y_v2/'
fpath_5y = f'{team_blob_url}/data/5yr_filtercols_historic_holidays.parquet' 
#f'{team_blob_url}/corrected_custom_join_5y/'
#df_3m = spark.read.parquet(fpath_3m)
df_5y = spark.read.parquet(fpath_5y)

# COMMAND ----------

display(df_5y)

# COMMAND ----------

#df_5y = df_5y.filter(col("year") != 2019)

# COMMAND ----------

row_count_5y = df_5y.count()
print(f"Number of rows in 12 month dataset: {row_count_5y}")

# COMMAND ----------

#row count before filtering out 2019
#row_count_12m = df_12m.count()
#print(f"Number of rows in 12 month dataset: {row_count_12m}")

# COMMAND ----------

#row_count_3m = df_3m.count()
#print(f"Number of rows in 3 month dataset: {row_count_3m}")

# COMMAND ----------

#df_12m = df_12m.dropDuplicates()
display(df_5y)

# COMMAND ----------

#row_count_12m = df_12m.count()
#print(f"Number of rows in 12 month dataset: {row_count_12m}")

# COMMAND ----------

split_wnd = F.split(F.col("WND"), ",")
split_cig = F.split(F.col("CIG"), ",")
split_vis = F.split(F.col("VIS"), ",")
split_tmp = F.split(F.col("TMP"), ",")
split_slp = F.split(F.col("SLP"), ",")
split_dew = F.split(F.col("DEW"), ",")

# Add separate columns for wind direction, wind speed, gust speed, ceiling height, visibility, temperature, and sea level pressure
df_new = df_5y.withColumn("wind_direction", 
                                   F.when(split_wnd[0] == "999", None).otherwise(split_wnd[0].cast("double"))) \
                        .withColumn("wind_speed", 
                                   F.when(split_wnd[3] == "999", None).otherwise((split_wnd[3].cast("double") / 10))) \
                        .withColumn("gust_speed", 
                                   F.when(split_wnd[4] == "999", None).otherwise(split_wnd[4].cast("double"))) \
                        .withColumn("ceiling_height", 
                                   F.when(split_cig[0] == "99999", None).otherwise(split_cig[0].cast("double"))) \
                        .withColumn("visibility", 
                                   F.when(split_vis[0] == "99999", None).otherwise((split_vis[0].cast("double") / 1000))) \
                        .withColumn("temperature", 
                                   F.when(split_tmp[0] == "999", None).otherwise((split_tmp[0].cast("double") / 10))) \
                        .withColumn("dew_point", 
                                   F.when(split_dew[0] == "999", None).otherwise((split_dew[0].cast("double") / 10))) \
                        .withColumn("sea_level_pressure", 
                                   F.when(split_slp[0] == "99999", None).otherwise((split_slp[0].cast("double") / 10)))

# Show the updated DataFrame
#df_with_new_columns.show(truncate=False)
display(df_new)

# COMMAND ----------

df_new = df_new.filter(~col("FL_DATE").startswith("2019"))
display(df_new)

# COMMAND ----------

row_count_5y = df_new.count()
print(f"Number of rows in 4 year dataset: {row_count_5y}")

# COMMAND ----------

def null_count(column_name): 
  null_count = df_new.filter(col(column_name).isNull()).count()
  print(f"Number of null values in {column_name}: {null_count}")

# COMMAND ----------

def eda(column_name):
  # Basic summary statistics: count, min, max, mean, stddev
  summary_stats = df_new.select(column_name).summary("count", "min", "max", "mean", "stddev")
  summary_stats.show()
  
  # Quartiles (using percentile_approx)
  quartiles = df_new.select(
      expr(f'percentile_approx({column_name}, 0.25)').alias('Q1'),
      expr(f'percentile_approx({column_name}, 0.5)').alias('Median'),
      expr(f'percentile_approx({column_name}, 0.75)').alias('Q3')
  )
  quartiles.show()

  # Calculate mean and standard deviation for outlier detection
  stats = df_new.select(mean(column_name).alias('mean'), stddev(column_name).alias('stddev')).first()
  mean_val = stats['mean']
  stddev_val = stats['stddev']
  
  # Outlier boundaries as ±3 standard deviations from the mean
  lower_bound = mean_val - 3 * stddev_val
  upper_bound = mean_val + 3 * stddev_val

  # Count outliers
  outliers = df_new.filter((col(column_name) < lower_bound) | (col(column_name) > upper_bound)).count()
  total_count = row_count_5y#df_12m.count()
  outlier_percentage = (outliers / total_count) * 100
  print(f"Percentage of outliers in {column_name}: {outlier_percentage}%")

  # Collect data for the specific column as a Pandas DataFrame
  column_data = df_new.select(column_name).filter((col(column_name) >= lower_bound) & (col(column_name) <= upper_bound)).dropna().toPandas()
  return column_data
  # Histogram
  #plt.figure(figsize=(10, 6))
  #plt.hist(column_data[column_name], bins=30, edgecolor='k', alpha=0.7)
  #plt.title(f'Histogram of {column_name}')
  #plt.xlabel(column_name)
  #plt.ylabel('Frequency')
  #plt.show()

  # Boxplot
  #plt.figure(figsize=(10, 6))
  #plt.boxplot(column_data[column_name], vert=False)
  #plt.title(f'Boxplot of {column_name}')
  #plt.xlabel(column_name)
  #plt.show()

# COMMAND ----------

is_delay_counts = df_new.groupBy("DEP_DEL15").count()
is_delay_counts.show()

# COMMAND ----------

is_cancelled = df_new.groupBy("CANCELLED").count()
is_cancelled.show()

# COMMAND ----------

delayed_canceled = df_new.filter((col('DEP_DEL15')==1) & (col('CANCELLED')==1)).count()
canceled_not_delayed = df_new.filter((col('DEP_DEL15')==0) & (col('CANCELLED')==1)).count()
delayed_canceled_is_null = df_new.filter((col('DEP_DEL15')==1) & (col('CANCELLED').isNull())).count()
delayed_not_canceled = df_new.filter((col('DEP_DEL15')==1) & (col('CANCELLED')==0)).count()

# Show results
print(f'\ndelayed_canceled: {delayed_canceled:,}')
print(f'canceled_not_delayed: {canceled_not_delayed:,}')
print(f'delayed_canceled_is_null: {delayed_canceled_is_null:,}')
print(f'delayed_not_canceled: {delayed_not_canceled:,}')

# COMMAND ----------

null_count("wind_direction")

# COMMAND ----------

null_count("wind_speed")

# COMMAND ----------

null_count("gust_speed")

# COMMAND ----------

null_count("ceiling_height")

# COMMAND ----------

null_count("visibility")

# COMMAND ----------

null_count("temperature")

# COMMAND ----------

null_count("dew_point")

# COMMAND ----------

null_count("sea_level_pressure")

# COMMAND ----------

col_data = eda("wind_direction")
plt.figure(figsize=(10, 6))
plt.hist(col_data["wind_direction"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Wind Direction')
plt.xlabel("Wind Direction (degrees)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

col_data = eda("wind_speed")
plt.figure(figsize=(10, 6))
plt.hist(col_data["wind_speed"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Wind Speed')
plt.xlabel("Wind Speed (knots)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

col_data = eda("gust_speed")
plt.figure(figsize=(10, 6))
plt.hist(col_data["gust_speed"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Gust Speed')
plt.xlabel("Gust Speed (knots)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

col_data = eda("ceiling_height")
plt.figure(figsize=(10, 6))
plt.hist(col_data["ceiling_height"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Cloud Ceiling Height')
plt.xlabel("Ceiling Height (meters)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

col_data = eda("visibility")
plt.figure(figsize=(10, 6))
plt.hist(col_data["visibility"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Visibility Distance')
plt.xlabel("Visibility Distance (kilometers)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

col_data = eda("temperature")
plt.figure(figsize=(10, 6))
plt.hist(col_data["temperature"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Temperature')
plt.xlabel("Temperature (celsius)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

col_data = eda("dew_point")
plt.figure(figsize=(10, 6))
plt.hist(col_data["dew_point"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Dew Point Temperature')
plt.xlabel("Dew Point Temperature (celsius)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

col_data = eda("sea_level_pressure")
plt.figure(figsize=(10, 6))
plt.hist(col_data["sea_level_pressure"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Sea Level Pressure')
plt.xlabel("Sea Level Pressure (hPa)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

null_count("CRS_ELAPSED_TIME")

# COMMAND ----------

null_count("DISTANCE")

# COMMAND ----------

col_data = eda("CRS_ELAPSED_TIME")
plt.figure(figsize=(10, 6))
plt.hist(col_data["CRS_ELAPSED_TIME"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Scheduled Flight Time')
plt.xlabel("Scheduled Flight Time (minutes)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

col_data = eda("DISTANCE")
plt.figure(figsize=(10, 6))
plt.hist(col_data["DISTANCE"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Flight Distance')
plt.xlabel("Flight Distance (miles)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# Calculate delayed flights per airport
delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 1)
    .groupBy("ORIGIN")
    .agg(count("*").alias("delayed_flights"))
)

# Calculate non-delayed flights per airport
non_delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 0)
    .groupBy("ORIGIN")
    .agg(count("*").alias("non_delayed_flights"))
)

# Combine both metrics
airport_flights = delayed_flights.join(
    non_delayed_flights, on="ORIGIN", how="outer"
).fillna(0)

# Convert to Pandas for visualization
airport_flights_pd = airport_flights.toPandas()

# Add total flights and select top 10 airports
airport_flights_pd["total_flights"] = (
    airport_flights_pd["delayed_flights"] + airport_flights_pd["non_delayed_flights"]
)
top_10_airports = airport_flights_pd.nlargest(10, "total_flights")

# Plot the data
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
x = np.arange(len(top_10_airports["ORIGIN"]))
width = 0.35

plt.figure(figsize=(12, 6))

plt.gca().set_facecolor("#212121")  # Axis background
plt.gcf().set_facecolor("#212121")  # Figure background

plt.bar(
    x - width / 2,
    top_10_airports["delayed_flights"],
    width,
    label="Delayed Flights",
    color="#F8AB41",
    edgecolor="black",
)
plt.bar(
    x + width / 2,
    top_10_airports["non_delayed_flights"],
    width,
    label="Non-Delayed Flights",
    color="#4DD0E1",
    edgecolor="black",
)

# Annotate bars with percentages
for i, (delayed, non_delayed) in enumerate(
    zip(top_10_airports["delayed_flights"], top_10_airports["non_delayed_flights"])
):
    total = delayed + non_delayed
    delayed_pct = (delayed / total) * 100 if total > 0 else 0
    non_delayed_pct = (non_delayed / total) * 100 if total > 0 else 0

    # Annotate delayed bars
    plt.text(
        x[i] - width / 2,
        delayed + 100,  # Offset above the bar
        f"{delayed_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
    )

    # Annotate non-delayed bars
    plt.text(
        x[i] + width / 2,
        non_delayed + 100,  # Offset above the bar
        f"{non_delayed_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
    )

# Customizing the plot
plt.title("Top 10 Airports: Delayed vs Non-Delayed Flights", fontsize=16, color="white")
plt.xlabel("Airport", fontsize=14, color="white")
plt.ylabel("Number of Flights", fontsize=14, color="white")
plt.xticks(x, top_10_airports["ORIGIN"], rotation=45, fontsize=12, color="white")
plt.yticks(fontsize=12, color="white")
plt.legend(fontsize=12, facecolor="#212121", edgecolor="white", labelcolor="white")
ax = plt.gca()  # Get current axis
for spine in ax.spines.values():
    spine.set_edgecolor("white")  # Set the color of each spine to white
ax.yaxis.get_offset_text().set_color("white")
plt.tight_layout()

plt.show()

# COMMAND ----------

# Calculate delayed flights per airport
delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 1)
    .groupBy("DAY_OF_WEEK")
    .agg(count("*").alias("delayed_flights"))
)

# Calculate non-delayed flights per airport
non_delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 0)
    .groupBy("DAY_OF_WEEK")
    .agg(count("*").alias("non_delayed_flights"))
)

# Combine both metrics
day_flights = delayed_flights.join(
    non_delayed_flights, on="DAY_OF_WEEK", how="outer"
).fillna(0)
day_order = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
}
# Convert to Pandas for visualization
day_flights_pd = day_flights.toPandas()

# Add total flights and select top 10 airports
day_flights_pd["total_flights"] = (
    day_flights_pd["delayed_flights"] + day_flights_pd["non_delayed_flights"]
)
day_flights_pd["DAY_OF_WEEK_NAME"] = day_flights_pd["DAY_OF_WEEK"].map(day_order)

# Sort by natural day order
day_flights_pd = day_flights_pd.sort_values(by="DAY_OF_WEEK")
# Plot the data
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
x = np.arange(len(day_flights_pd["DAY_OF_WEEK"]))
width = 0.35

plt.figure(figsize=(12, 6))

plt.gca().set_facecolor("#212121")  # Axis background
plt.gcf().set_facecolor("#212121")  # Figure background

plt.bar(
    x - width / 2,
    day_flights_pd["delayed_flights"],
    width,
    label="Delayed Flights",
    color="#F8AB41",
    edgecolor="black",
)
plt.bar(
    x + width / 2,
    day_flights_pd["non_delayed_flights"],
    width,
    label="Non-Delayed Flights",
    color="#4DD0E1",
    edgecolor="black",
)

# Annotate bars with percentages
for i, (delayed, non_delayed) in enumerate(
    zip(day_flights_pd["delayed_flights"], day_flights_pd["non_delayed_flights"])
):
    total = delayed + non_delayed
    delayed_pct = (delayed / total) * 100 if total > 0 else 0
    non_delayed_pct = (non_delayed / total) * 100 if total > 0 else 0

    # Annotate delayed bars
    plt.text(
        x[i] - width / 2,
        delayed + 100,  # Offset above the bar
        f"{delayed_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
    )

    # Annotate non-delayed bars
    plt.text(
        x[i] + width / 2,
        non_delayed + 100,  # Offset above the bar
        f"{non_delayed_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
    )

# Customizing the plot
plt.title("Delayed vs Non-Delayed Flights by Weekday", fontsize=16, color="white")
plt.xlabel("Day of Week", fontsize=14, color="white")
plt.ylabel("Number of Flights", fontsize=14, color="white")
plt.xticks(x, day_flights_pd["DAY_OF_WEEK"], rotation=45, fontsize=12, color="white")
plt.yticks(fontsize=12, color="white")
plt.legend(fontsize=12, facecolor="#212121", edgecolor="white", labelcolor="white")
ax = plt.gca()  # Get current axis
for spine in ax.spines.values():
    spine.set_edgecolor("white")  # Set the color of each spine to white
ax.yaxis.get_offset_text().set_color("white")
plt.tight_layout()

plt.show()

# COMMAND ----------

# Calculate delayed flights per airport
delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 1)
    .groupBy("OP_UNIQUE_CARRIER")
    .agg(count("*").alias("delayed_flights"))
)

# Calculate non-delayed flights per airport
non_delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 0)
    .groupBy("OP_UNIQUE_CARRIER")
    .agg(count("*").alias("non_delayed_flights"))
)

# Combine both metrics
airline_flights = delayed_flights.join(
    non_delayed_flights, on="OP_UNIQUE_CARRIER", how="outer"
).fillna(0)

# Convert to Pandas for visualization
airline_flights_pd = airline_flights.toPandas()

# Add total flights and select top 10 airports
airline_flights_pd["total_flights"] = (
    airline_flights_pd["delayed_flights"] + airline_flights_pd["non_delayed_flights"]
)
top_10_airlines = airline_flights_pd.nlargest(10, "total_flights")

# Plot the data
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
x = np.arange(len(top_10_airlines["OP_UNIQUE_CARRIER"]))
width = 0.35

plt.figure(figsize=(12, 6))

# Set background color
plt.gca().set_facecolor("#212121")  # Axis background
plt.gcf().set_facecolor("#212121")  # Figure background

# Bars for delayed and non-delayed flights
delayed_bars = plt.bar(
    x - width / 2,
    top_10_airlines["delayed_flights"],
    width,
    label="Delayed Flights",
    color="#F8AB41",
    edgecolor="black",
)
non_delayed_bars = plt.bar(
    x + width / 2,
    top_10_airlines["non_delayed_flights"],
    width,
    label="Non-Delayed Flights",
    color="#4DD0E1",
    edgecolor="black",
)

# Annotate bars with percentages
for i, (delayed, non_delayed) in enumerate(
    zip(top_10_airlines["delayed_flights"], top_10_airlines["non_delayed_flights"])
):
    total = delayed + non_delayed
    delayed_pct = (delayed / total) * 100 if total > 0 else 0
    non_delayed_pct = (non_delayed / total) * 100 if total > 0 else 0

    # Annotate delayed bars
    plt.text(
        x[i] - width / 2,
        delayed + 100,  # Offset above the bar
        f"{delayed_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
    )

    # Annotate non-delayed bars
    plt.text(
        x[i] + width / 2,
        non_delayed + 100,  # Offset above the bar
        f"{non_delayed_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
    )

# Customizing the plot
plt.title("Top 10 Airlines: Delayed vs Non-Delayed Flights", fontsize=16, color="white")
plt.xlabel("Airlines", fontsize=14, color="white")
plt.ylabel("Number of Flights", fontsize=14, color="white")
plt.xticks(x, top_10_airlines["OP_UNIQUE_CARRIER"], rotation=45, fontsize=12, color="white")
plt.yticks(fontsize=12, color="white")
plt.legend(fontsize=12, facecolor="#212121", edgecolor="white", labelcolor="white")
ax = plt.gca()  # Get current axis
for spine in ax.spines.values():
    spine.set_edgecolor("white")  # Set the color of each spine to white

ax.yaxis.get_offset_text().set_color("white")
plt.tight_layout()

# Show the plot
plt.show()


# COMMAND ----------

# Calculate delayed flights per airport
delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 1)
    .groupBy("MONTH")
    .agg(count("*").alias("delayed_flights"))
)

# Calculate non-delayed flights per airport
non_delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 0)
    .groupBy("MONTH")
    .agg(count("*").alias("non_delayed_flights"))
)

# Combine both metrics
month_flights = delayed_flights.join(
    non_delayed_flights, on="MONTH", how="outer"
).fillna(0)
month_order = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}
# Convert to Pandas for visualization
month_flights_pd = month_flights.toPandas()

# Add total flights and select top 10 airports
month_flights_pd["total_flights"] = (
    month_flights_pd["delayed_flights"] + month_flights_pd["non_delayed_flights"]
)
month_flights_pd["MONTH_NAME"] = month_flights_pd["MONTH"].map(month_order)

# Sort by natural day order
month_flights_pd = month_flights_pd.sort_values(by="MONTH")
# Plot the data
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
x = np.arange(len(month_flights_pd["MONTH"]))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(
    x - width / 2,
    month_flights_pd["delayed_flights"],
    width,
    label="Delayed Flights",
    color="red",
    edgecolor="black",
)
plt.bar(
    x + width / 2,
    month_flights_pd["non_delayed_flights"],
    width,
    label="Non-Delayed Flights",
    color="green",
    edgecolor="black",
)

# Customizing the plot
plt.title("Delayed vs Non-Delayed Flights by Month", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Number of Flights", fontsize=14)
plt.xticks(x, month_flights_pd["MONTH"], rotation=45, fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()

# COMMAND ----------

# Calculate delayed flights per airport
delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 1)
    .groupBy("DEP_TIME_BLK")
    .agg(count("*").alias("delayed_flights"))
)

# Calculate non-delayed flights per airport
non_delayed_flights = (
    df_new.filter(col("DEP_DEL15") == 0)
    .groupBy("DEP_TIME_BLK")
    .agg(count("*").alias("non_delayed_flights"))
)

# Combine both metrics
hour_flights = delayed_flights.join(
    non_delayed_flights, on="DEP_TIME_BLK", how="outer"
).fillna(0)
hour_order = {
    1: "0100-0159",
    2: "0200-0259",
    3: "0300-0359",
    4: "0400-0459",
    5: "0500-0559",
    6: "0600-0659",
    7: "0700-0759",
    8: "0800-0859",
    9: "0900-0959",
    10: "1000-1059",
    11: "1100-1159",
    12: "1200-1259",
    13: "1300-1359",
    14: "1400-1459",
    15: "1500-1559",
    16: "1600-1659",
    17: "1700-1759",
    18: "1800-1859",
    19: "1900-1959",
    20: "2000-2059",
    21: "2100-2159",
    22: "2200-2259",
    23: "2300-2359",
    24: "2400-2459"
}
# Convert to Pandas for visualization
hour_flights_pd = hour_flights.toPandas()

# Add total flights and select top 10 airports
hour_flights_pd["total_flights"] = (
    hour_flights_pd["delayed_flights"] + hour_flights_pd["non_delayed_flights"]
)
hour_flights_pd["Hour_Name"] = hour_flights_pd["DEP_TIME_BLK"].map(hour_order)
# Sort by natural day order
hour_flights_pd = hour_flights_pd.sort_values(by="DEP_TIME_BLK")
# Plot the data
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
x = np.arange(len(hour_flights_pd["DEP_TIME_BLK"]))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(
    x - width / 2,
    hour_flights_pd["delayed_flights"],
    width,
    label="Delayed Flights",
    color="red",
    edgecolor="black",
)
plt.bar(
    x + width / 2,
    hour_flights_pd["non_delayed_flights"],
    width,
    label="Non-Delayed Flights",
    color="green",
    edgecolor="black",
)

# Customizing the plot
plt.title("Delayed vs Non-Delayed Flights by Hour", fontsize=16)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Number of Flights", fontsize=14)
plt.xticks(x, hour_flights_pd["DEP_TIME_BLK"], rotation=45, fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()

# COMMAND ----------

# Define bin intervals
bin_size = 10
max_wind_speed = 50

# Add a column for wind speed bins
df_binned = df_new.withColumn(
    "wind_speed_bin",
    when(col("wind_speed").isNotNull(), 
         floor(col("wind_speed") / bin_size) * bin_size).cast("int")
).filter(col("wind_speed_bin").isNotNull() & (col("wind_speed_bin") <= max_wind_speed))

# Calculate delayed flights per bin
delayed_flights = (
    df_binned.filter(col("DEP_DEL15") == 1)
    .groupBy("wind_speed_bin")
    .agg(count("*").alias("delayed_flights"))
)

# Calculate non-delayed flights per bin
non_delayed_flights = (
    df_binned.filter(col("DEP_DEL15") == 0)
    .groupBy("wind_speed_bin")
    .agg(count("*").alias("non_delayed_flights"))
)

# Combine both metrics
wind_flights = delayed_flights.join(
    non_delayed_flights, on="wind_speed_bin", how="outer"
).fillna(0)
wind_flights = wind_flights.orderBy("wind_speed_bin")
# Convert to Pandas for visualization
wind_flights_pd = wind_flights.toPandas()

# Plotting
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
x = np.arange(len(wind_flights_pd["wind_speed_bin"]))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(
    x - width / 2,
    wind_flights_pd["delayed_flights"],
    width,
    label="Delayed Flights",
    color="red",
    edgecolor="black",
)
plt.bar(
    x + width / 2,
    wind_flights_pd["non_delayed_flights"],
    width,
    label="Non-Delayed Flights",
    color="green",
    edgecolor="black",
)

# Customizing the plot
plt.title("Delayed vs Non-Delayed Flights by Wind Speed Bins", fontsize=16)
plt.xlabel("Wind Speed (mph)", fontsize=14)
plt.ylabel("Number of Flights", fontsize=14)
plt.xticks(x, wind_flights_pd["wind_speed_bin"].astype(str) + "-" + 
              (wind_flights_pd["wind_speed_bin"] + bin_size).astype(str), rotation=45, fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()


# COMMAND ----------

#df_new.write.parquet(f"{team_blob_url}/data/5yr_parsed_historic_holidays.parquet", mode='overwrite')


# COMMAND ----------

#display(dbutils.fs.ls(f"{team_blob_url}/data"))

# COMMAND ----------

df_new = spark.read.parquet(f"{team_blob_url}/data/5yr-feat.parquet")
display(df_new)

# COMMAND ----------

null_count("ceiling_height_is_below_10000")

# COMMAND ----------

null_count("ceiling_height_is_between_10000_20000")

# COMMAND ----------

null_count("ceiling_height_is_above_20000")

# COMMAND ----------

null_count("isHoliday")

# COMMAND ----------

null_count("specWeather")

# COMMAND ----------

row_count_5y = df_new.count()
print(f"Number of rows in 4 year dataset: {row_count_5y}")

# COMMAND ----------

null_count("origin_yr_flights")

# COMMAND ----------

null_count("HIST_DEP_FLT_NUM")

# COMMAND ----------

null_count("HIST_ARR_FLT_NUM")

# COMMAND ----------

col_data = eda("origin_yr_flights")
plt.figure(figsize=(10, 6))
plt.hist(col_data["origin_yr_flights"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Flights per Year')
plt.xlabel("Flights per Year (Origin)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

null_count("origin_yr_flights")

# COMMAND ----------

null_count("dest_yr_flights")

# COMMAND ----------

col_data = eda("dest_yr_flights")
plt.figure(figsize=(10, 6))
plt.hist(col_data["dest_yr_flights"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Flights per Year')
plt.xlabel("Flights per Year (Dest)")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

null_count("HIST_ARR_DELAY")

# COMMAND ----------

col_data = eda("HIST_ARR_DELAY")
plt.figure(figsize=(10, 6))
plt.hist(col_data["HIST_ARR_DELAY"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Historical Arrival Delay')
plt.xlabel("Historical Arrival Delay")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

null_count("HIST_DEP_DELAY")

# COMMAND ----------

col_data = eda("HIST_DEP_DELAY")
plt.figure(figsize=(10, 6))
plt.hist(col_data["HIST_DEP_DELAY"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of Historical Departure Delay')
plt.xlabel("Historical Departure Delay")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

null_count("pagerank")

# COMMAND ----------

col_data = eda("pagerank")
plt.figure(figsize=(10, 6))
plt.hist(col_data["pagerank"], bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Histogram of pagerank')
plt.xlabel("pagerank")
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

null_count("pagerank")