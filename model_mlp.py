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
from pyspark.sql.functions import row_number, when, concat#, #sum, row_number, concat, to_timestamp
from pyspark.sql.functions import isnan, when, count, col, split, trim, lit, avg, lpad
from pyspark.sql.functions import expr, mean, stddev, randn, to_date, round, upper, trim, countDistinct

# Spark ML Lib functions
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors

# Feature pipeline
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer, OneHotEncoder, MinMaxScaler, VectorAssembler, Imputer, SQLTransformer, StandardScaler

# Regression model
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import lit, col, row_number, monotonically_increasing_id, rand
from sklearn.model_selection import TimeSeriesSplit

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import random

import optuna
from optuna.pruners import MedianPruner

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

display(dbutils.fs.ls(f"{team_blob_url}"))

# COMMAND ----------

#fpath = f'{team_blob_url}/data/1yr_preprocessed_no.parquet'#f'{team_blob_url}/data/1yr_folds.parquet'
FPATH = f'{team_blob_url}/data/5yr_preprocessed.parquet'
df = spark.read.parquet(FPATH)

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.filter("pagerank is not null")

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.filter(df.DEP_DEL15.isNotNull())

# Create a combined column for fold and split
df = df.withColumn(
    "fold_split",
    concat(col("foldCol"), lit("_"), col("split"))
)

# COMMAND ----------

class_counts = df.groupBy("DEP_DEL15").count().collect()
class_counts_dict = {row["DEP_DEL15"]: row["count"] for row in class_counts}
print(class_counts_dict[0])
print(class_counts_dict[1])

# COMMAND ----------

balanced_df = df

# COMMAND ----------

balanced_df.groupBy("DEP_DEL15").count().show()

# COMMAND ----------

display(balanced_df)

# COMMAND ----------

balanced_df.groupBy("foldCol", "split").count().orderBy("foldCol", "split").show()

# COMMAND ----------


# Enumerate columns for ohe ("categorical"), already ohe ("ohe") and for scaling ()
need_ohe_cols = ['QUARTER', 'MONTH', 'DAY_OF_WEEK','OP_UNIQUE_CARRIER', 'DEP_TIME_BLK', 'ARR_TIME_BLK', 'DAY_OF_MONTH']
already_ohe_cols = ['ceiling_height_is_below_10000', 'ceiling_height_is_between_10000_20000', 'ceiling_height_is_above_20000']
standard_scalar_cols = ['wind_direction', 'temperature', 'sea_level_pressure'] # wind, temp and pressure
min_max_cols = ['CRS_ELAPSED_TIME', 'DISTANCE',
                'visibility', 'dew_point', 'wind_speed', 'gust_speed' # vis, dew, weather and wind
                ]

# Consolidate
label_col = 'DEP_DEL15'
categorical_cols = need_ohe_cols + already_ohe_cols

# Adding augmented features
need_ohe_cols = need_ohe_cols + ['HIST_ARR_FLT_NUM', 'HIST_DEP_FLT_NUM'] # number of flights in the past
already_ohe_cols = already_ohe_cols + ['isHoliday','specWeather']
min_max_cols = min_max_cols + ['HIST_ARR_DELAY', 'HIST_DEP_DELAY', 'origin_yr_flights', 'dest_yr_flights']    
numerical_cols = min_max_cols + standard_scalar_cols + ['pagerank'] # PageRank (which is null for yr1)

# Removing extra time columns
need_ohe_cols = [c for c in need_ohe_cols if c not in ['QUARTER', 'MONTH', 'DAY_OF_WEEK', 'DAY_OF_MONTH']]

# COMMAND ----------

df_train_filtered = balanced_df.dropDuplicates()

# COMMAND ----------


indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in need_ohe_cols]
encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_ohe", dropLast=False) for col in need_ohe_cols]
imputer = Imputer(strategy="mean", inputCols=numerical_cols, outputCols=[f"{col}_imputed" for col in numerical_cols])

minmax_assembler = VectorAssembler(inputCols=[f"{col}_imputed" for col in min_max_cols], outputCol="minmax_features")
minmax_scaler = MinMaxScaler(inputCol="minmax_features", outputCol="scaled_minmax_features")

# Assemble input columns for StandardScaler
standard_assembler = VectorAssembler(inputCols=[f"{col}_imputed" for col in standard_scalar_cols], outputCol="standard_features")
standard_scaler = StandardScaler(inputCol="standard_features", outputCol="scaled_standard_features", withMean=True, withStd=True)

final_assembler = VectorAssembler(
        inputCols=[f"{col}_ohe" for col in need_ohe_cols] + already_ohe_cols + ["scaled_minmax_features"] + ["scaled_standard_features"],
        outputCol="final_features"
    )

pipeline =  Pipeline(stages=indexers + encoders + [imputer] + [minmax_assembler, minmax_scaler, standard_assembler, standard_scaler, final_assembler])

pipeline_model = pipeline.fit(df_train_filtered)
df_encoded = pipeline_model.transform(df_train_filtered)

# COMMAND ----------

display(df_encoded)

# COMMAND ----------

input_size = df_encoded.select("final_features").rdd.map(lambda row: row[0].size).distinct().collect()

# COMMAND ----------

input_size = input_size[0]
print(input_size)

# COMMAND ----------

def get_train_val_data(df, fold):
    """
    Function to filter training and validation sets
    """
    train_data = df.filter((col('split') == 'train') & (col('foldCol') == fold))
    val_data = df.filter((col('split') == 'val') & (col('foldCol') == fold))
    return train_data, val_data

def objective(trial):
    expected_length = input_size
    hidden_layer_size = trial.suggest_int("hidden_layer_size", 10, expected_length * 2)
    max_iter = trial.suggest_int("max_iter", 10, 100)
    block_size = 128#trial.suggest_int("block_size", expected_length / 4, expected_length, log=True)
    step_size = trial.suggest_float("step_size", 0.01, 0.1, log=True)
    
    layers = [expected_length, hidden_layer_size, 2]
    mlp = MultilayerPerceptronClassifier(
        featuresCol="final_features",
        labelCol=label_col,
        maxIter=max_iter,
        blockSize=block_size,
        stepSize = step_size,
        layers=layers
    )
    
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)

    fold_metrics = []

    # Define number of folds
    num_folds = df_encoded.select('foldCol').distinct().count()
    print("Num folds:", num_folds)
    # Loop through folds
    for fold in range(num_folds):
        
        # Get train and validation data
        train_data, val_data = get_train_val_data(df_encoded, fold)

        # Train the pipeline and evaluate on both train and validation
        try:
            print(fold)
            pipeline_model = mlp.fit(train_data)
            print("A")
            training_preds = pipeline_model.transform(train_data)
            print("B")
            val_preds = pipeline_model.transform(val_data)
            print("C")
            predictions = {"train": training_preds, "val": val_preds}
            print("D")
        except Exception as e:
            trial.set_user_attr("failure_reason", str(e))
            raise optuna.exceptions.TrialPruned()
        
        # Store metrics per key
        train_val = []
        # Calculate training and validation metrics for this fold
        for key, prediction in predictions.items():
            tp = prediction.filter((col(label_col) == 1) & (col("prediction") == 1)).count()
            fp = prediction.filter((col(label_col) == 0) & (col("prediction") == 1)).count()
            fn = prediction.filter((col(label_col) == 1) & (col("prediction") == 0)).count()
            tn = prediction.filter((col(label_col) == 0) & (col("prediction") == 0)).count()
            print("true positives: ", tp)
            print("false positives: ", fp)
            print("true negatives: ", tn)
            print("fasle negatives: ", fn)

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print("precision: ", precision)
            print("recall: ", recall)
            # F-beta score with beta=0.5
            beta = 0.5
            fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
            print("fbeta: ", fbeta)
            # Store metrics for this fold
            p_key = "precision_" + key
            r_key = "recall_" + key
            fb_key = "fbeta_" + key

            train_val.append({
                p_key: precision,
                r_key: recall,
                fb_key: fbeta
            })
        
        fold_weight = {
            "fold": fold,
            "weight": fold + 1  # Add weight to the fold metrics
        }
        
        # Merge together and append to fold metrics
        fold_metrics.append((fold_weight | train_val[0] | train_val[1]))

    # Aggregate metrics across folds with weights
    total_weight = sum([m["weight"] for m in fold_metrics])
    avg_metrics = {
        metric: sum([m[metric] * m["weight"] for m in fold_metrics]) / total_weight
        for metric in ["precision_train", "recall_train", "fbeta_train", "precision_val", "recall_val", "fbeta_val"]
    }

    # Log metrics to Optuna trial
    for metric_name, metric_value in avg_metrics.items():
        trial.set_user_attr(f"weighted_average_{metric_name}", metric_value)

    # Report primary metric for pruning
    trial.report(avg_metrics["fbeta_val"], step=0)

    # Prune trial if not promising
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return avg_metrics["fbeta_val"]

# COMMAND ----------

# MAGIC %md
# MAGIC ##5 YEAR TRAINING

# COMMAND ----------

study = optuna.create_study(direction="maximize", 
                            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1
                                                )
                            )
study.optimize(objective, 
               n_trials=10, 
               n_jobs=-1
               )

# COMMAND ----------

metrics_df = study.trials_dataframe()
metrics_df.T

# COMMAND ----------

display(metrics_df.loc[0, 'user_attrs_failure_reason'])

# COMMAND ----------

optuna.visualization.matplotlib.plot_timeline(study)

# COMMAND ----------

optuna.visualization.matplotlib.plot_optimization_history(study)

# COMMAND ----------

optuna.visualization.matplotlib.plot_contour(study)

# COMMAND ----------

optuna.visualization.matplotlib.plot_intermediate_values(study)

# COMMAND ----------

optuna.visualization.matplotlib.plot_terminator_improvement(study)

# COMMAND ----------

optuna.visualization.plot_parallel_coordinate(study)

# COMMAND ----------

ps.DataFrame(metrics_df).to_csv(f"{team_blob_url}/results/5yr_mlp1", index=True)

# COMMAND ----------

study_name = "baseline_5yr_xval_study"

# Save the study object locally as a pickle file
local_pickle_path = f"/tmp/{study_name}.pkl"

with open(local_pickle_path, "wb") as f:
    pickle.dump(study, f)

print(f"Study saved locally to {local_pickle_path}")

# Use Spark to upload the pickle file to Azure Blob Storage
dbutils.fs.cp(f"file://{local_pickle_path}", f"{team_blob_url}/data/{study_name}.pkl")
print(f"Study uploaded to {team_blob_url}/data/{study_name}.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ##RERUN TRAINING MODEL WITH BEST HYPERPARAMETERS TO GET METRICS ACROSS FOLDS

# COMMAND ----------

def get_train_val_data(df, fold):
    """
    Function to filter training and validation sets
    """
    train_data = df.filter((col('split') == 'train') & (col('foldCol') == fold))
    val_data = df.filter((col('split') == 'val') & (col('foldCol') == fold))
    return train_data, val_data

def objective(trial):
    input_size = 85
    mlp = MultilayerPerceptronClassifier(
        featuresCol="final_features",
        labelCol=label_col,
        maxIter=88,#best_params["max_iter"],#89,
        blockSize=128,#best_params["block_size"],#24,
        stepSize = 0.015321856604164128,#best_params["step_size"],#0.012987,
        layers=[input_size, 
                21,#best_params["hidden_layer_1_size"], 
                2]
    )
    fold_metrics = {"train": [], "val": []}
    num_folds = df_encoded.select('foldCol').distinct().count()
    fold_weights = range(1, num_folds + 1)

    for fold in range(num_folds):
        train_data, val_data = get_train_val_data(df_encoded, fold)
        print(f"Fold {fold} of {num_folds}")
        pipeline_model = mlp.fit(train_data)
       
        # Evaluate on train and validation data
        for split_name, data in [("train", train_data), ("val", val_data)]:
            predictions = pipeline_model.transform(data)
            #auc = evaluator.evaluate(predictions)
            print('a')
            # Compute metrics
            tp = predictions.filter((col(label_col) == 1) & (col("prediction") == 1)).count()
            fp = predictions.filter((col(label_col) == 0) & (col("prediction") == 1)).count()
            fn = predictions.filter((col(label_col) == 1) & (col("prediction") == 0)).count()
            tn = predictions.filter((col(label_col) == 0) & (col("prediction") == 0)).count()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            f1_micro = accuracy
            f1_macro = (precision + recall) / 2 if (precision + recall) > 0 else 0
            f1_weighted = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            beta = 0.5
            fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
            print('b')
            # Store fold metrics
            fold_metrics[split_name].append({
                "fold": fold,
                #"auc": auc,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "fbeta": fbeta,
                "weight": fold_weights[fold]
            })

            # Log fold-specific metrics
            for metric_name, metric_value in {
                "precision": precision, "recall": recall,
                "accuracy": accuracy, "f1_micro": f1_micro, "f1_macro": f1_macro,
                "f1_weighted": f1_weighted, "fbeta": fbeta
            }.items():
                trial.set_user_attr(f"{split_name}_fold_{fold}_{metric_name}", metric_value)

    # Aggregate metrics across folds for train and val separately
    metrics_per_split = {}
    for split_name in ["train", "val"]:
        total_weight = sum(m["weight"] for m in fold_metrics[split_name])

        metrics_per_split[split_name] = {
            metric: sum(m[metric] * m["weight"] for m in fold_metrics[split_name]) / total_weight
            for metric in ["precision", "recall", "accuracy", "f1_micro", "f1_macro", "f1_weighted", "fbeta"]
        }

    # Log aggregated metrics
    for split_name, metrics in metrics_per_split.items():
        for metric_name, metric_value in metrics.items():
            trial.set_user_attr(f"{split_name}_weighted_average_{metric_name}", metric_value)

    # Store per-fold metrics for later analysis
    trial.set_user_attr("fold_metrics", fold_metrics)

    trial.report(metrics_per_split["val"]["fbeta"], step=0)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    print('Fold metrics stored in trial user attributes')
    print('val fbeta:', metrics_per_split["val"]["fbeta"])
    return metrics_per_split["val"]["fbeta"]

study = optuna.create_study(direction="maximize", 
                            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1
                                                )
                            )
study.optimize(objective, 
               n_trials=1, 
               n_jobs=-1
               )


# COMMAND ----------

def study_to_dataframe(study):
    records = []
    for trial in study.trials:
        # Extract general metrics
        record = {
            "trial_number": trial.number,
            "state": trial.state.name,  # Use human-readable state
            "train_accuracy": trial.user_attrs.get("train_weighted_average_accuracy"),
            #"train_auc": trial.user_attrs.get("train_weighted_average_auc"),
            "train_f1_macro": trial.user_attrs.get("train_weighted_average_f1_macro"),
            "train_f1_micro": trial.user_attrs.get("train_weighted_average_f1_micro"),
            "train_f1_weighted": trial.user_attrs.get("train_weighted_average_f1_weighted"),
            "train_fbeta": trial.user_attrs.get("train_weighted_average_fbeta"),
            "train_precision": trial.user_attrs.get("train_weighted_average_precision"),
            "train_recall": trial.user_attrs.get("train_weighted_average_recall"),
            "val_accuracy": trial.user_attrs.get("val_weighted_average_accuracy"),
            #"val_auc": trial.user_attrs.get("val_weighted_average_auc"),
            "val_f1_macro": trial.user_attrs.get("val_weighted_average_f1_macro"),
            "val_f1_micro": trial.user_attrs.get("val_weighted_average_f1_micro"),
            "val_f1_weighted": trial.user_attrs.get("val_weighted_average_f1_weighted"),
            "val_fbeta": trial.user_attrs.get("val_weighted_average_fbeta"),
            "val_precision": trial.user_attrs.get("val_weighted_average_precision"),
            "val_recall": trial.user_attrs.get("val_weighted_average_recall"),
            **trial.params,  # Add hyperparameters as columns
        }

        # Add fold-specific metrics
        for key, value in trial.user_attrs.items():
            if "fold_" in key:  # Capture fold-specific metrics
                record[key] = value

        records.append(record)

    # Create a DataFrame with pyspark.pandas
    df_metrics = ps.DataFrame(records)
    return df_metrics


# Convert study results to a DataFrame
df_metrics = study_to_dataframe(study)

# COMMAND ----------

display(df_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ##5 YEAR TEST

# COMMAND ----------

def calculate_metrics(prediction, beta=0.5):
    tp = prediction.filter((col(label_col) == 1) & (col("prediction") == 1)).count()
    fp = prediction.filter((col(label_col) == 0) & (col("prediction") == 1)).count()
    fn = prediction.filter((col(label_col) == 1) & (col("prediction") == 0)).count()
    tn = prediction.filter((col(label_col) == 0) & (col("prediction") == 0)).count()

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F-beta score with beta=0.5
    beta = 0.5
    fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0

    return fbeta, precision, recall

# COMMAND ----------

best_params = study.best_params

# COMMAND ----------

def blind_test(train, test):
    mlp = MultilayerPerceptronClassifier(
        featuresCol="final_features",
        labelCol=label_col,
        maxIter=best_params["max_iter"],
        blockSize=128,#best_params["block_size"],
        stepSize = best_params["step_size"],
        layers=[input_size, best_params["hidden_layer_size"], 2]
    )
    print("Max Iter: ", best_params["max_iter"])
    print("Step Size: ", best_params["step_size"])
    print("Hidden Layer Size: ", best_params["hidden_layer_size"])
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in need_ohe_cols]
    encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_ohe", dropLast=False) for col in need_ohe_cols]
    imputer = Imputer(strategy="mean", inputCols=numerical_cols, outputCols=[f"{col}_imputed" for col in numerical_cols])

    minmax_assembler = VectorAssembler(inputCols=[f"{col}_imputed" for col in min_max_cols], outputCol="minmax_features")
    minmax_scaler = MinMaxScaler(inputCol="minmax_features", outputCol="scaled_minmax_features")

    # Assemble input columns for StandardScaler
    standard_assembler = VectorAssembler(inputCols=[f"{col}_imputed" for col in standard_scalar_cols], outputCol="standard_features")
    standard_scaler = StandardScaler(inputCol="standard_features", outputCol="scaled_standard_features", withMean=True, withStd=True)

    final_assembler = VectorAssembler(
            inputCols=[f"{col}_ohe" for col in need_ohe_cols] + already_ohe_cols + ["scaled_minmax_features"] + ["scaled_standard_features"],
            outputCol="final_features"
        )
    
    pipeline =  Pipeline(stages=indexers + encoders + [imputer] + [minmax_assembler, minmax_scaler, standard_assembler, standard_scaler, final_assembler, mlp])

    mlp_model = pipeline.fit(train)
    training_preds = mlp_model.transform(train)
    test_preds = mlp_model.transform(test)

    predictions = {"train": training_preds, "test": test_preds}

    # Store metrics per key
    train_test = []
    # Calculate training and validation metrics
    print("A")
    for key, prediction in predictions.items():
        fbeta, precision, recall = calculate_metrics(prediction)

        # Store metrics for this fold
        p_key = "precision_" + key
        r_key = "recall_" + key
        fb_key = "fbeta_" + key

        train_test.append({
            p_key: precision,
            r_key: recall,
            fb_key: fbeta
        })
    
    metrics = train_test[0] | train_test[1]

    return metrics

# COMMAND ----------

train_5yr = df_train_filtered

# COMMAND ----------

display(train_5yr)

# COMMAND ----------

test_5yr = spark.read.parquet(f'{team_blob_url}/data/5yr_test.parquet')

# COMMAND ----------

display(test_5yr)

# COMMAND ----------


test_5yr = test_5yr.filter(test_5yr.DEP_DEL15.isNotNull())
test_5yr = test_5yr.withColumn("specWeather", F.when(col("specWeather") > 0, 1).otherwise(0))

# COMMAND ----------

test_5yr = test_5yr.drop(*['ORIGIN', 'DEST','FL_DATE', 'CANCELLED', 'YEAR'])

# COMMAND ----------

results = blind_test(train_5yr, test_5yr)

# COMMAND ----------

display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ##FEATURE IMPORTANCE

# COMMAND ----------

input_size = 85
mlp = MultilayerPerceptronClassifier(
    featuresCol="final_features",
    labelCol=label_col,
    maxIter=best_params["max_iter"],
    blockSize=128,#best_params["block_size"],
    stepSize = best_params["step_size"],
    layers=[input_size, 
            best_params["hidden_layer_size"], 
            2]
)
print("Max Iter: ", best_params["max_iter"])
print("Step Size: ", best_params["step_size"])
print("Hidden Layer 1 Size: ", best_params["hidden_layer_size"])
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in need_ohe_cols]
encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_ohe", dropLast=False) for col in need_ohe_cols]
imputer = Imputer(strategy="mean", inputCols=numerical_cols, outputCols=[f"{col}_imputed" for col in numerical_cols])

minmax_assembler = VectorAssembler(inputCols=[f"{col}_imputed" for col in min_max_cols], outputCol="minmax_features")
minmax_scaler = MinMaxScaler(inputCol="minmax_features", outputCol="scaled_minmax_features")

# Assemble input columns for StandardScaler
standard_assembler = VectorAssembler(inputCols=[f"{col}_imputed" for col in standard_scalar_cols], outputCol="standard_features")
standard_scaler = StandardScaler(inputCol="standard_features", outputCol="scaled_standard_features", withMean=True, withStd=True)

final_assembler = VectorAssembler(
        inputCols=[f"{col}_ohe" for col in need_ohe_cols] + already_ohe_cols + ["scaled_minmax_features"] + ["scaled_standard_features"],
        outputCol="final_features"
    )

pipeline =  Pipeline(stages=indexers + encoders + [imputer] + [minmax_assembler, minmax_scaler, standard_assembler, standard_scaler, final_assembler, mlp])

mlp_model = pipeline.fit(train_5yr.filter(col('split') == 'train'))
mlp_model = mlp_model.stages[-1]  

# COMMAND ----------

coefficients = mlp_model.weights.toArray()

feature_names = categorical_cols + numerical_cols 
feature_coeffs = dict(zip(feature_names, coefficients))

# Calculate class-specific contributions
class_1_contributions = coefficients  # Coefficients directly favor Class 1
class_0_contributions = -coefficients  # Negative coefficients favor Class 0

# Sort coefficients for Class 1
sorted_class_1 = sorted(zip(feature_names, class_1_contributions), key=lambda x: abs(x[1]), reverse=True)
sorted_names_1, sorted_values_1 = zip(*sorted_class_1)

# Sort coefficients for Class 0
sorted_class_0 = sorted(zip(feature_names, class_0_contributions), key=lambda x: abs(x[1]), reverse=True)
sorted_names_0, sorted_values_0 = zip(*sorted_class_0)

# Define styling for both plots
bar_color = "#4DD0E1"
background_color = "#212121"
label_color = "white"

# Plot Coefficients for Class 1
plt.figure(figsize=(10, 6))
plt.barh(sorted_names_1, sorted_values_1, color=bar_color)
plt.xlabel("Coefficient Value for Class 1", color=label_color)
plt.ylabel("Features", color=label_color)
plt.title("MLP 1 Hidden Layer Coefficients (Class 1)", color=label_color)
plt.gca().invert_yaxis()
plt.gca().set_facecolor(background_color)
plt.gcf().set_facecolor(background_color)
plt.tick_params(colors=label_color)
plt.grid(False)
plt.show()

# Plot Coefficients for Class 0
plt.figure(figsize=(10, 6))
plt.barh(sorted_names_0, sorted_values_0, color=bar_color)
plt.xlabel("Coefficient Value for Class 0", color=label_color)
plt.ylabel("Features", color=label_color)
plt.title("MLP 1 Hidden Layer Coefficients (Class 0)", color=label_color)
plt.gca().invert_yaxis()
plt.gca().set_facecolor(background_color)
plt.gcf().set_facecolor(background_color)
plt.tick_params(colors=label_color)
plt.grid(False)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##1 YEAR EXPERIMENT

# COMMAND ----------

study = optuna.create_study(direction="maximize", 
                            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1
                                                )
                            )
study.optimize(objective, 
               n_trials=5, 
               n_jobs=-1
               )

# COMMAND ----------

metrics_df = study.trials_dataframe()
metrics_df.T

# COMMAND ----------

def get_train_val_data(df, fold):
    """
    Function to filter training and validation sets
    """
    train_data = df.filter((col('split') == 'train') & (col('foldCol') == fold))
    val_data = df.filter((col('split') == 'val') & (col('foldCol') == fold))
    return train_data, val_data

def objective(trial):
    expected_length = 125
    hidden_layer_size = trial.suggest_int("hidden_layer_size", 10, expected_length * 2)
    max_iter = trial.suggest_int("max_iter", 10, 100)
    block_size = trial.suggest_int("block_size", 50, 120, log=True)
    step_size = trial.suggest_float("step_size", 0.01, 0.1, log=True)
    imputation_strategy = trial.suggest_categorical("imputation_strategy", ["mean", "median"])

    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in need_ohe_cols]
    encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_ohe", dropLast=False) for col in need_ohe_cols]
    imputer = Imputer(strategy=imputation_strategy, inputCols=numerical_cols, outputCols=[f"{col}_imputed" for col in numerical_cols])

    minmax_assembler = VectorAssembler(inputCols=min_max_cols, outputCol="minmax_features")
    minmax_scaler = MinMaxScaler(inputCol="minmax_features", outputCol="scaled_minmax_features")

    # Assemble input columns for StandardScaler
    standard_assembler = VectorAssembler(inputCols=standard_scalar_cols, outputCol="standard_features")
    standard_scaler = StandardScaler(inputCol="standard_features", outputCol="scaled_standard_features", withMean=True, withStd=True)

    final_assembler = VectorAssembler(
            inputCols=[f"{col}_ohe" for col in need_ohe_cols] + already_ohe_cols + ["scaled_minmax_features"] + ["scaled_standard_features"],
            outputCol="final_features"
        )    

    #temp_pipeline = Pipeline(stages=indexers + encoders + [imputer] + [minmax_assembler, minmax_scaler, standard_assembler, standard_scaler, final_assembler])
    #temp_model = temp_pipeline.fit(test_df)
    #df_temp = temp_model.transform(test_df)
    #expected_length = df_temp.select("final_features").rdd.map(lambda row: row[0].size).distinct().collect()
    #display(df_temp)
    # Determine the sparse vector length
    #first_row = df_temp.select("final_features").first()
    #expected_length = first_row["final_features"].size
    #print(expected_length)

    #hidden_layer_size = trial.suggest_int("hidden_layer_size", 10, expected_length)
    layers = [expected_length, hidden_layer_size, 2]
    mlp = MultilayerPerceptronClassifier(
        featuresCol="final_features",
        labelCol=label_col,
        maxIter=max_iter,
        blockSize=block_size,
        stepSize = step_size,
        layers=layers
    )
    pipeline =  Pipeline(stages=indexers + encoders + [imputer] + [minmax_assembler, minmax_scaler, standard_assembler, standard_scaler, final_assembler, mlp])
    
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)

    fold_metrics = []

    # Define number of folds
    num_folds = test_df.select('foldCol').distinct().count()
    print("Num folds:", num_folds)
    # Loop through folds
    for fold in range(num_folds):
        
        # Get train and validation data
        train_data, val_data = get_train_val_data(test_df, fold)

        # Train the pipeline and evaluate on both train and validation
        try:
            print("0")
            pipeline_model = pipeline.fit(train_data)
            print("A")
            training_preds = pipeline_model.transform(train_data)
            print("B")
            val_preds = pipeline_model.transform(val_data)
            print("C")
            predictions = {"train": training_preds, "val": val_preds}
            print("D")
        except Exception as e:
            trial.set_user_attr("failure_reason", str(e))
            raise optuna.exceptions.TrialPruned()
        
        # Store metrics per key
        train_val = []
        # Calculate training and validation metrics for this fold
        for key, prediction in predictions.items():
            tp = prediction.filter((col(label_col) == 1) & (col("prediction") == 1)).count()
            fp = prediction.filter((col(label_col) == 0) & (col("prediction") == 1)).count()
            fn = prediction.filter((col(label_col) == 1) & (col("prediction") == 0)).count()
            tn = prediction.filter((col(label_col) == 0) & (col("prediction") == 0)).count()

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # F-beta score with beta=0.5
            beta = 0.5
            fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0

            # Store metrics for this fold
            p_key = "precision_" + key
            r_key = "recall_" + key
            fb_key = "fbeta_" + key

            train_val.append({
                p_key: precision,
                r_key: recall,
                fb_key: fbeta
            })
        
        fold_weight = {
            "fold": fold,
            "weight": fold + 1  # Add weight to the fold metrics
        }
        
        # Merge together and append to fold metrics
        fold_metrics.append((fold_weight | train_val[0] | train_val[1]))

    # Aggregate metrics across folds with weights
    total_weight = sum([m["weight"] for m in fold_metrics])
    avg_metrics = {
        metric: sum([m[metric] * m["weight"] for m in fold_metrics]) / total_weight
        for metric in ["precision_train", "recall_train", "fbeta_train", "precision_val", "recall_val", "fbeta_val"]
    }

    # Log metrics to Optuna trial
    for metric_name, metric_value in avg_metrics.items():
        trial.set_user_attr(f"weighted_average_{metric_name}", metric_value)

    # Report primary metric for pruning
    trial.report(avg_metrics["fbeta_val"], step=0)

    # Prune trial if not promising
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return avg_metrics["fbeta_val"]

# COMMAND ----------

display(metrics_df.loc[0, 'user_attrs_failure_reason'])

# COMMAND ----------

def objective(trial):
    hidden_layer_size = trial.suggest_int("hidden_layer_size", 10, 125)
    max_iter = trial.suggest_int("max_iter", 10, 100)
    block_size = trial.suggest_int("block_size", 64, 256, log=True)
    step_size = trial.suggest_float("step_size", 0.01, 0.1, log=True)
    imputation_strategy = trial.suggest_categorical("imputation_strategy", ["mean", "median"])

    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_ohe", dropLast=False) for col in categorical_cols]
    imputer = Imputer(strategy=imputation_strategy, inputCols=numerical_cols, outputCols=[f"{col}_imputed" for col in numerical_cols])

    minmax_assembler = VectorAssembler(inputCols=min_max_cols, outputCol="minmax_features")
    minmax_scaler = MinMaxScaler(inputCol="minmax_features", outputCol="scaled_minmax_features")

    # Assemble input columns for StandardScaler
    standard_assembler = VectorAssembler(inputCols=standard_scalar_cols + ohe_cols, outputCol="standard_features")
    standard_scaler = StandardScaler(inputCol="standard_features", outputCol="scaled_standard_features", withMean=True, withStd=True)

    final_assembler = VectorAssembler(
            inputCols=[f"{col}_ohe" for col in categorical_cols] + ["scaled_minmax_features"] + ["scaled_standard_features"],
            outputCol="final_features"
        )    
    temp_pipeline = Pipeline(stages=indexers + encoders + [imputer] + [minmax_assembler, minmax_scaler, standard_assembler, standard_scaler, final_assembler])
    temp_model = temp_pipeline.fit(df_train_filtered)
    df_temp = temp_model.transform(df_train_filtered)

    # Determine the sparse vector length
    first_row = df_temp.select("final_features").first()
    expected_length = first_row["final_features"].size
    print(expected_length)
    layers = [expected_length, hidden_layer_size, 2]
    mlp = MultilayerPerceptronClassifier(
        featuresCol="final_features",
        labelCol=label_col,
        maxIter=max_iter,
        blockSize=block_size,
        stepSize = step_size,
        layers=layers
    )
    pipeline =  Pipeline(stages=indexers + encoders + [imputer] + [minmax_assembler, minmax_scaler, standard_assembler, standard_scaler, final_assembler, mlp])

    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    paramGrid = ParamGridBuilder() \
    .addGrid(mlp.maxIter, [10, 50, 100]) \
    .addGrid(mlp.blockSize, [64, 128, 256]) \
    .addGrid(mlp.stepSize, [0.01, 0.05, 0.1]) \
    .addGrid(mlp.layers, [[expected_length, expected_length / 4, 2], [expected_length, expected_length / 2, 2], [expected_length, expected_length, 2]]) \
    .build()

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=5,
        foldCol="foldCol",
        parallelism=1
    )

    cv_model = crossval.fit(df_train_filtered)

    # Generate predictions
    predictions = cv_model.transform(df_train_filtered)

    # Initialize metrics dictionary
    metrics = {}

    # Binary classification metrics
    metrics["auc_roc"] = evaluator.evaluate(predictions)

    # Custom F-beta calculation
    tp = predictions.filter((col(label_col) == 1) & (col("prediction") == 1)).count()
    fp = predictions.filter((col(label_col) == 0) & (col("prediction") == 1)).count()
    fn = predictions.filter((col(label_col) == 1) & (col("prediction") == 0)).count()
    tn = predictions.filter((col(label_col) == 0) & (col("prediction") == 0)).count()

    # Calculate metrics
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1-score calculations
    metrics["f1_micro"] = metrics["accuracy"]  # Equivalent for micro-averaging in binary classification
    metrics["f1_macro"] = (metrics["precision"] + metrics["recall"]) / 2 if (metrics["precision"] + metrics["recall"]) > 0 else 0
    metrics["f1_weighted"] = (2 * metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0

    # F-beta score with beta=0.5
    beta = 0.5
    if metrics["precision"] + metrics["recall"] == 0:
        metrics["fbeta"] = 0
    else:
        metrics["fbeta"] = (1 + beta**2) * (metrics["precision"] * metrics["recall"]) / ((beta**2 * metrics["precision"]) + metrics["recall"])

    # Log metrics to Optuna trial
    for metric_name, metric_value in metrics.items():
        trial.set_user_attr(metric_name, metric_value)

    # Report primary metric for pruning
    trial.report(metrics["fbeta"], step=0)

    # Prune trial if not promising
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return metrics["fbeta"]

# COMMAND ----------

import pandas as pd

# Convert Optuna study to a DataFrame
def study_to_dataframe(study):
    records = []
    for trial in study.trials:
        record = {
            "trial_number": trial.number,
            "state": trial.state,
            "accuracy": trial.user_attrs.get("accuracy"),
            "precision": trial.user_attrs.get("precision"),
            "recall": trial.user_attrs.get("recall"),
            "f1_micro": trial.user_attrs.get("f1_micro"),
            "f1_macro": trial.user_attrs.get("f1_macro"),
            "f1_weighted": trial.user_attrs.get("f1_weighted"),
            "auc_roc": trial.user_attrs.get("auc_roc"),
            **trial.params,
        }
        records.append(record)
    return pd.DataFrame(records)

# Create the DataFrame
df_metrics = study_to_dataframe(study)
df_metrics.T


# COMMAND ----------

optuna.visualization.matplotlib.plot_timeline(study)

# COMMAND ----------

optuna.visualization.matplotlib.plot_optimization_history(study)

# COMMAND ----------

optuna.visualization.matplotlib.plot_contour(study)

# COMMAND ----------

optuna.visualization.matplotlib.plot_intermediate_values(study)

# COMMAND ----------

optuna.visualization.matplotlib.plot_terminator_improvement(study)

# COMMAND ----------

optuna.visualization.plot_parallel_coordinate(study)