import os
import sys
import numpy as np
import pandas as pd

"""
Defining common constant variables for the training pipeline
"""
TARGET_COLUMN: str = "high_traffic"
PIPELINE_NAME: str = "RecipeSiteTraffic"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "recipe_site_traffic.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"""
Data ingestion related constants start with DATA_INGESTION var name
"""

DATA_INGESTION_COLLECTION_NAME: str = "traffic_data"
DATA_INGESTION_DATABASE_NAME: str = "recipe_db"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data validation related constants start with DATA_VALIDATION var name
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str ="valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.json"
DATA_VALIDATION_SCHEMA_REPORT_DIR: str = "schema_report"
DATA_VALIDATION_SCHEMA_REPORT_FILE_NAME: str = "schema_report.json"


"""
Data transformation related constants start with DATA_TRANSFORMATION var name
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_PREPROCESSED_DATA_DIR: str = "preprocessed_data"
DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_DIR: str = "preprocessor_object"
DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_FILE_NAME: str = "preprocessor.joblib"