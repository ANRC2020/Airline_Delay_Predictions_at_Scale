# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

import datetime

import pandas as pd
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.functions import when

# COMMAND ----------

# Mount blob storage and grant team access
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

# Spark Context
sc = spark.sparkContext
spark

# COMMAND ----------

# # The following can write the dataframe to the team's Cloud Storage  
# # Navigate back to your Storage account in https://portal.azure.com, to inspect the partitions/files.
# df.write.parquet(f"{team_blob_url}/EDA")

# # see what's in the blob storage root folder 
# display(dbutils.fs.ls(f"{team_blob_url}"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Joined Dataframe and Select Relevant Columns

# COMMAND ----------

# Change to relevant path
DF_PATH = f'{team_blob_url}/corrected_custom_join_5y/'
#DF_PATH = f"{team_blob_url}/data/5yr_filtercols_historic.parquet"

# COMMAND ----------

df_raw = spark.read.parquet(DF_PATH)

# COMMAND ----------


features = ['QUARTER', 'MONTH', 'DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'DEP_TIME_BLK',  'ARR_TIME_BLK', 'CRS_ELAPSED_TIME',  'DISTANCE', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP']

label = ['DEP_DEL15']

other_uses = ["YEAR", "FL_DATE", 'DAY_OF_MONTH', 'TAIL_NUM', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CANCELLED', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_DELAY', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME'] # to extract other features - drop later

maybe_add = ['CANCELLED',  'DIVERTED', 'FLIGHTS', 'OP_CARRIER_FL_NUM'] # add section in report on why not including these

aw = ['AW1', 'AW2', 'AW3', 'AW4', 'AW5', 'AW6', 'AW7'] # Automated weather observation systems detecting different weather phenomena like rain, snow, or specific weather types.

mw = ['MW1', 'MW2', 'MW3', 'MW4', 'MW5', 'MW6'] # Miscellaneous weather phenomena such as mist, fog, or smoke.

ids = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'FL_DATE', 'CRS_DEP_TIME'] # if needed for joining
 

# COMMAND ----------

cols_to_keep = features + label + other_uses + aw + mw
filtered_df = df_raw.select(cols_to_keep)
display(filtered_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Parsing Weather Data

# COMMAND ----------

weather_cols = ["WND", "CIG", "VIS", "TMP", "SLP", "DEW"]

split_wnd = F.split(F.col("WND"), ",")
split_cig = F.split(F.col("CIG"), ",")
split_vis = F.split(F.col("VIS"), ",")
split_tmp = F.split(F.col("TMP"), ",")
split_slp = F.split(F.col("SLP"), ",")
split_dew = F.split(F.col("DEW"), ",")

# Add separate columns for wind direction, wind speed, gust speed, ceiling height, visibility, temperature, dew point, and sea level pressure
parsed_df = filtered_df.withColumn("wind_direction", 
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
                        .withColumn("sea_level_pressure", 
                                   F.when(split_slp[0] == "99999", None).otherwise((split_slp[0].cast("double") / 10))) \
                        .withColumn("dew_point", 
                                   F.when(split_dew[0] == "999", None).otherwise((split_dew[0].cast("double") / 10)))

parsed_df=parsed_df.drop(*weather_cols)
display(parsed_df)

# COMMAND ----------

parsed_df = (
    parsed_df
    .withColumn("ceiling_height_is_below_10000", when(col("ceiling_height") < 10000, 1).otherwise(0))
    .withColumn("ceiling_height_is_between_10000_20000", 
                when((col("ceiling_height") >= 10000) & (col("ceiling_height") < 20000), 1).otherwise(0))
    .withColumn("ceiling_height_is_above_20000", when(col("ceiling_height") >= 20000, 1).otherwise(0))
)
display(parsed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Checkpoint for Baseline
# MAGIC

# COMMAND ----------

parsed_df.write.parquet(f"{team_blob_url}/baseline/new/5yr_raw.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Add Special Weather Indicator

# COMMAND ----------

spec_cols = aw + mw
spec_df = parsed_df.withColumn('specWeather', sum([(~(F.isnan(c) | col(c).isNull() | (col(c) == ""))).cast("int") for c in spec_cols]))

# COMMAND ----------

# spec_df.where(col("specWeather") != 0).count() / spec_df.count()

# COMMAND ----------

# drop extra weather features
spec_df = spec_df.drop(*spec_cols)
display(spec_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Get Average Number of Flights for Origin and Destination

# COMMAND ----------

# remove test data
test_year = 2019
df_4yr = spec_df.select(["ORIGIN", "YEAR"]).where(col("YEAR") != test_year).withColumnRenamed('ORIGIN', 'airport')

# average flights per airport per year
num_years = df_4yr.select(col("YEAR")).distinct().count()
airport_volume = df_4yr.select(["airport"]).groupBy("airport").agg((F.count("*")/num_years).alias("avg_yr_flights")).sort(col("avg_yr_flights").desc())


# COMMAND ----------

# join origin and airport volume
join_df = spec_df.join(airport_volume, spec_df.ORIGIN == airport_volume.airport, "leftouter").withColumnRenamed('avg_yr_flights', 'origin_yr_flights').drop('airport')

# COMMAND ----------

# join dest and airport volume
airports_df = join_df.join(airport_volume, join_df.DEST == airport_volume.airport,"leftouter").withColumnRenamed('avg_yr_flights', 'dest_yr_flights').drop('airport')

# COMMAND ----------

display(airports_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Add Holiday Indicator Feature

# COMMAND ----------

# Azure storage access info
blob_account_name = "azureopendatastorage"
blob_container_name = "holidaydatacontainer"
blob_relative_path = "Processed"
blob_sas_token = r""

# COMMAND ----------

# Allow SPARK to read from Blob remotely
wasbs_path = 'wasbs://%s@%s.blob.core.windows.net/%s' % (blob_container_name, blob_account_name, blob_relative_path)
spark.conf.set(
  'fs.azure.sas.%s.%s.blob.core.windows.net' % (blob_container_name, blob_account_name),
  blob_sas_token)
print('Remote blob path: ' + wasbs_path)

# COMMAND ----------

# SPARK read parquet, note that it won't load any data yet by now
df = spark.read.parquet(wasbs_path)
print('Register the DataFrame as a SQL temporary view: source')
df.createOrReplaceTempView('source')

# COMMAND ----------

df = df.filter(col("countryOrRegion") == lit("United States"))

# COMMAND ----------

df.show(10)

# COMMAND ----------

start_date = datetime.datetime.strptime("2015-01-01", "%Y-%m-%d")
end_date = datetime.datetime.strptime("2019-12-31", "%Y-%m-%d")
df = df.filter(col("date").between(start_date, end_date))
df.show(10)

# COMMAND ----------

df.where(col("isPaidTimeOff").isNull()).show()

# COMMAND ----------

df.where(col("isPaidTimeOff") == lit("false")).show()

# COMMAND ----------

df = df.filter(col("isPaidTimeOff") != lit("false"))

# COMMAND ----------

df = df.select("date")

# COMMAND ----------

def get_range(date, offset):
    full_range = np.arange(-offset, offset+1)
    return [date + datetime.timedelta(days=int(x)) for x in full_range]

# COMMAND ----------

offset = 3
df = df.withColumn("range", F.array(get_range(col("date"), offset)))

# COMMAND ----------

holidays = df.select("range").rdd.flatMap(lambda x:x).flatMap(lambda x: x).collect()

# COMMAND ----------

# Convert flight date column type from string to date
hol_df = airports_df.withColumn("date_timestamp", F.to_date(airports_df["FL_DATE"], 'yyyy-MM-dd'))

# COMMAND ----------

# Add feature column
hol_df = hol_df.withColumn("isHoliday", F.when(col("date_timestamp").isin(holidays), 1).otherwise(0))
hol_df = hol_df.drop("date_timestamp")
display(hol_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Add PageRank feature

# COMMAND ----------

hol_df = hol_df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### read in pre-saved dataframe

# COMMAND ----------

pageRank_df = spark.read.parquet(f"{team_blob_url}/data/graph/graph_5yr.parquet").cache()

# COMMAND ----------

display(pageRank_df.sort("YEAR_NEW", "QUARTER_NEW", "pagerank", ascending=[True, True, False]))

# COMMAND ----------

df_missing = pageRank_df.filter(F.col("id").isNull() | F.col("pagerank").isNull() | F.col("YEAR").isNull() | F.col("QUARTER").isNull())

# Display the rows with missing data
display(df_missing)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join page rank to data features

# COMMAND ----------

# pageRank_df = pageRank_df.cache()
pgrank_df = hol_df.join(
    pageRank_df,
    (hol_df["YEAR"] == pageRank_df["YEAR_NEW"]) & (hol_df["QUARTER"] == pageRank_df["QUARTER_NEW"]) & (hol_df["ORIGIN"] == pageRank_df["id"]),
    "left"
).cache()
pgrank_df = pgrank_df.drop("YEAR_NEW", "QUARTER_NEW","id")
display(pgrank_df)
# pgrank_df = ...

# COMMAND ----------

display(pgrank_df.filter(col("QUARTER") == 2))

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Add Lag features

# COMMAND ----------

# MAGIC %md
# MAGIC ### Functions

# COMMAND ----------

# MAGIC %md
# MAGIC #### Lag features

# COMMAND ----------

from pyspark.sql.types import TimestampType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from pyspark.sql.functions import col,isnan, when, count, col, split, trim, lit, avg, sum
from pyspark.sql.functions import lpad, col, concat, lit, to_timestamp
from pyspark.sql.functions import expr
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, lag, lead

def create_hist_lag_vars(df,total_lags,columns_to_lag = ["ARR_DATETIME_UTC","ARR_DELAY"],isArrivalData = True):
    """
        Create historical lagging variables using the following approach.
                    1). Create timestamp from previous departure time in local time (PREV_DEP_LOCAL_*).
                    2). Convert previous departure time to UTC using latitude and longitude (PREV_DEP_UTC_*).
                    3). Add actual elapsed time to departure time (PREV_ARR_UTC_*).
                    4). Subtract previous arrival time from current scheduled departure (PREV_ARR_TIME_DIFF_*).
        INPUT: df - spark dataframe.
               num_lags - integer for number of lags.
               columns_to_lag - list of columns to create lagging variable.
               isArrivalData - boolean.
        OUTPUT: df - cleaned spark dataframe.
    """
    ## if we are working with arrival lagging data
    if isArrivalData:
        w = Window().partitionBy(col("TAIL_NUM")).orderBy(col("CRS_DEP_DATETIME_UTC"))
        # df = df.sort(col("TAIL_NUM").asc(),col("sched_depart_date_time_UTC").asc()).cache()
        ## only going back one day
        for num_lags in range(1,total_lags+1):
            for col_ in columns_to_lag:
                df = df.select("*", lag(col_,num_lags).over(w).alias(f"PREV_{col_}_{num_lags}"))
        return df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Time differencing

# COMMAND ----------

def diff_in_minutes(ts1, ts2):
    if ts1 is None or ts2 is None:
        return None
    diff = (ts1 - ts2).total_seconds() / 60
    return int(diff)
# Register the UDF
diff_in_minutes_udf = udf(diff_in_minutes, IntegerType())

# COMMAND ----------

# MAGIC %md
# MAGIC ### UTC Offset

# COMMAND ----------

file = 'custom_5y_utc_lookup'
df_utc = spark.read.parquet(f'{team_blob_url}/{file}/')
# rename columns.
df_utc = df_utc.withColumnRenamed('ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_ID_UTC')
# drop columns.
df_utc = df_utc.drop('ORIGIN_CITY_NAME','UTC_offset')
# rename columns.
df_utc = df_utc.withColumnRenamed('UTC_offset_hours', 'UTC_offset')
display(df_utc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join UTC with dataframe

# COMMAND ----------

pgrank_df_utc = pgrank_df.join(
    df_utc,
    (pgrank_df["ORIGIN_AIRPORT_ID"] == df_utc["ORIGIN_AIRPORT_ID_UTC"]),
    "left"
)
pgrank_df_utc = pgrank_df_utc.drop('ORIGIN_AIRPORT_ID_UTC')
display(pgrank_df_utc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create UTC time for departure and arrival

# COMMAND ----------

# create local scheduled departure time.
pgrank_df_utc = pgrank_df_utc.withColumn(f"CRS_DEP_TIME_PADDED", lpad(col(f"CRS_DEP_TIME"), 4, '0')) \
   .withColumn(f"CRS_DEP_DATETIME_STR", concat(col(f"FL_DATE"), lit(":"), col(f"CRS_DEP_TIME_PADDED"))) \
   .withColumn(f"CRS_DEP_DATETIME_LOCAL", to_timestamp(col(f"CRS_DEP_DATETIME_STR"), "yyyy-MM-dd:HHmm"))

# # convert to UTC by adding UTC offset column.
pgrank_df_utc = pgrank_df_utc.withColumn("CRS_DEP_DATETIME_UTC", expr("CRS_DEP_DATETIME_LOCAL + INTERVAL 1 HOUR * UTC_offset"))

# create actual arrival time in UTC
## - Arrival time (UTC) = Scheduled departure (UTC) + DEP_DELAY + ARR_DELAY + ACTUAL_ELAPSED TIME
pgrank_df_utc = pgrank_df_utc.withColumn("ARR_DATETIME_UTC", expr("CRS_DEP_DATETIME_UTC + INTERVAL 1 MINUTE * (DEP_DELAY + ARR_DELAY + ACTUAL_ELAPSED_TIME)"))

# create actual departure time in UTC
## -  Actual departure time (UTC) = Scheduled departure (UTC) + DEP_DELAY
pgrank_df_utc = pgrank_df_utc.withColumn("DEP_DATETIME_UTC", expr("CRS_DEP_DATETIME_UTC + INTERVAL 1 MINUTE * (DEP_DELAY)"))

display(pgrank_df_utc.orderBy('TAIL_NUM', 'CRS_DEP_DATETIME_UTC') \
                   .filter(col("CANCELLED") == 0)
                   )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create lagged features

# COMMAND ----------

cols_to_lag = ["ARR_DATETIME_UTC","ARR_DELAY","DEP_DATETIME_UTC","DEP_DELAY"]
lag_df = create_hist_lag_vars(pgrank_df_utc, 3, columns_to_lag = cols_to_lag)

display(lag_df.orderBy('TAIL_NUM', 'CRS_DEP_DATETIME_UTC'))

# COMMAND ----------


# difference previous arrivals with scheduled departure.
for num_lags in range(1, 4):
  lag_df = lag_df.withColumn(f"PREV_ARR_TIME_DIFF_{num_lags}",diff_in_minutes_udf(col("CRS_DEP_DATETIME_UTC").cast(TimestampType()), col(f"PREV_ARR_DATETIME_UTC_{num_lags}").cast(TimestampType())))

# difference previous departures with scheduled departure.
for num_lags in range(1, 4):
  lag_df = lag_df.withColumn(f"PREV_DEP_TIME_DIFF_{num_lags}",diff_in_minutes_udf(col("CRS_DEP_DATETIME_UTC").cast(TimestampType()), col(f"PREV_DEP_DATETIME_UTC_{num_lags}").cast(TimestampType())))

# create conditional based on 120 minute window for both departure and arrival.
## arrival
lag_df = lag_df.withColumn(
    "HIST_ARR_DELAY",
    F.when(col("PREV_ARR_TIME_DIFF_1") >= 120, col("PREV_ARR_DELAY_1"))
     .when((col("PREV_ARR_TIME_DIFF_1") < 120) & (col("PREV_ARR_TIME_DIFF_2") >= 120), col("PREV_ARR_DELAY_2"))
     .otherwise(col("PREV_ARR_DELAY_3"))
)

lag_df = lag_df.withColumn(
    "HIST_ARR_FLT_NUM",
    F.when(col("PREV_ARR_TIME_DIFF_1") >= 120, 1)
     .when((col("PREV_ARR_TIME_DIFF_1") < 120) & (col("PREV_ARR_TIME_DIFF_2") >= 120), 2)
     .otherwise(3)
)
## departure
lag_df = lag_df.withColumn(
    "HIST_DEP_DELAY",
    F.when(col("PREV_DEP_TIME_DIFF_1") >= 120, col("PREV_DEP_DELAY_1"))
     .when((col("PREV_DEP_TIME_DIFF_1") < 120) & (col("PREV_DEP_TIME_DIFF_2") >= 120), col("PREV_DEP_DELAY_2"))
     .otherwise(col("PREV_DEP_DELAY_3"))
)

lag_df = lag_df.withColumn(
    "HIST_DEP_FLT_NUM",
    F.when(col("PREV_DEP_TIME_DIFF_1") >= 120, 1)
     .when((col("PREV_DEP_TIME_DIFF_1") < 120) & (col("PREV_DEP_TIME_DIFF_2") >= 120), 2)
     .otherwise(3)
)

# drop intermediate columns.
columns_to_drop = ["PREV_ARR_DATETIME_UTC_1","PREV_ARR_DELAY_1",
                  "PREV_ARR_DATETIME_UTC_2","PREV_ARR_DELAY_2",
                  "PREV_ARR_DATETIME_UTC_3","PREV_ARR_DELAY_3",
                  "PREV_ARR_TIME_DIFF_1","PREV_ARR_TIME_DIFF_2","PREV_ARR_TIME_DIFF_3"
                  ]
lag_df = lag_df.drop(*columns_to_drop)

columns_to_drop = ["PREV_DEP_DATETIME_UTC_1","PREV_DEP_DELAY_1",
                  "PREV_DEP_DATETIME_UTC_2","PREV_DEP_DELAY_2",
                  "PREV_DEP_DATETIME_UTC_3","PREV_DEP_DELAY_3",
                  "PREV_DEP_TIME_DIFF_1","PREV_DEP_TIME_DIFF_2","PREV_DEP_TIME_DIFF_3"
                  ]

lag_df = lag_df.drop(*columns_to_drop).cache()

# COMMAND ----------

display(lag_df.orderBy('TAIL_NUM', 'CRS_DEP_DATETIME_UTC') \
                 .filter(col("CANCELLED") == 0))

# COMMAND ----------

lag_df.columns

# COMMAND ----------

# extra columns.
columns_to_drop = ["CRS_DEP_DATETIME_STR","CRS_DEP_DATETIME_LOCAL","CRS_DEP_DATETIME_UTC","ARR_DATETIME_UTC","DEP_DATETIME_UTC",'ACTUAL_ELAPSED_TIME',"CRS_DEP_TIME","CRS_ARR_TIME","DEP_DELAY","ARR_DELAY","TAIL_NUM","ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID","UTC_offset","ceiling_height"]
lag_df = lag_df.drop(*columns_to_drop).cache()

# COMMAND ----------

lag_df.columns

# COMMAND ----------

lag_df = lag_df.drop("CRS_DEP_TIME_PADDED")

# COMMAND ----------

# lag_df = ...
display(lag_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Checkpoint new dataframe as parquet

# COMMAND ----------

# Drop extra columns
extra_cols = []
feat_df = lag_df.drop(extra_cols)

# COMMAND ----------

lag_df.write.parquet(f"{team_blob_url}/data/5yr-feat.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC # Graveyard

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code for generating page rank

# COMMAND ----------


import networkx as nx
from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import *
from graphframes.examples import Graphs
from pyspark.sql.types import StructType,StringType

seasonal_var = 'QUARTER'

if seasonal_var == 'QUARTER':
    seasonal_thresh = 4
elif seasonal_var == 'MONTH':
    seasonal_thresh = 12

count = 0
# for year in hol_df.select('YEAR').distinct().collect():
for yr in [2015,2016,2017,2018,2019]:
    # yr = int(year.YEAR)
    print(f'Year :{yr}')
    for quarter in hol_df.select(seasonal_var).distinct().collect():
        quart = int(quarter[seasonal_var])
        print(f'Quarter: {quart}')
        edges_df = hol_df.filter(hol_df.YEAR == yr) \
                                      .filter(hol_df[seasonal_var] == quart) \
                                      .select(["ORIGIN", "DEST"]) \
                                      .withColumn("relationship",lit("flights")) \
                                      .withColumnRenamed("ORIGIN", "src") \
                                      .withColumnRenamed("DEST", "dst").cache()
        # Get distinct values in the "ORIGIN" column
        origin_vals = edges_df.select("src").distinct().rdd.flatMap(lambda x: x).collect()

        # Get distinct values in the "DEST" column
        dest_vals = edges_df.select("dst").distinct().rdd.flatMap(lambda x: x).collect()

        # Combine both lists and remove duplicates
        all_airports = set(origin_vals).union(set(dest_vals))

        # Convert the set to a list of tuples
        all_airports_tuples = [(airport,) for airport in all_airports]

        # Create pyspark dataframe for airports
        schema = StructType().add("id", StringType())
        df_airport_unique = spark.createDataFrame(all_airports_tuples, schema).cache()

        # create graph.
        g = GraphFrame(df_airport_unique, edges_df)

        # page rank.
        results = g.pageRank(resetProbability=0.15, tol=0.01)
        # add in seasonal variables.
        pr_df = results.vertices
        pr_df = pr_df.withColumn('YEAR',lit(yr)) \
                     .withColumn(seasonal_var,lit(quart)).cache()

        if count == 0:
            pr_df_final = pr_df
        else:
            pr_df_final = pr_df_final.union(pr_df)

        count += 1
    ## checkpoint yearly data
    pr_df_final.write.parquet(f"{team_blob_url}/data/graph/graph_{yr}.parquet")
    print(f'finished checkpointing {yr}')




# COMMAND ----------

# MAGIC %md
# MAGIC ### read each yearly file and concat

# COMMAND ----------

count = 0
for yr in [2015,2016,2017,2018,2019]:
    test_pr = spark.read.parquet(f"{team_blob_url}/data/graph/graph_{yr}.parquet/")
    if count == 0:
        pageRank_df = test_pr
    else:
        pageRank_df = pageRank_df.union(test_pr)
    count += 1
pageRank_df = pageRank_df.cache()
# pr_df_final.write.parquet(f"{team_blob_url}/data/graph/graph_5yr.parquet")
pageRank_df.write.format("parquet").mode("overwrite").save(f"{team_blob_url}/data/graph/graph_5yr.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ### reindex quarter to next quarter to prevent leakage

# COMMAND ----------

seasonal_var = "QUARTER"
seasonal_thresh = 4
pageRank_df = pageRank_df.withColumn(f"{seasonal_var}_NEW", when(col(seasonal_var) < seasonal_thresh, col(seasonal_var) + 1).otherwise(col(seasonal_var) - (seasonal_thresh-1))) \
                                 .withColumn("YEAR_NEW", when(col(seasonal_var) < seasonal_thresh, col("YEAR")).otherwise(col("YEAR") + 1))
print('Done w/ seasonality leakage')

# drop columns.
pageRank_df = pageRank_df.drop(seasonal_var, "YEAR")

# cache.
pageRank_df = pageRank_df.cache()