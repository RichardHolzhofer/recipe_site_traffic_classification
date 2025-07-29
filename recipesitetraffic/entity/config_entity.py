import os
from datetime import datetime
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.constants import constants

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%d_%m_%Y_%H_%M_%S")
        self.pipeline_name = constants.PIPELINE_NAME
        self.artifact_name = constants.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp = timestamp
        

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            constants.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_FEATURE_STORE_DIR,
            constants.FILE_NAME           
        )
        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_INGESTED_DIR,
            constants.TRAIN_FILE_NAME
        )
        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_INGESTED_DIR,
            constants.TEST_FILE_NAME
        )
        self.train_test_split_ratio = constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name = constants.DATA_INGESTION_COLLECTION_NAME
        self.database_name = constants.DATA_INGESTION_DATABASE_NAME
        
        
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            constants.DATA_VALIDATION_DIR_NAME
            )
        self.valid_data_dir = os.path.join(
            self.data_validation_dir,
            constants.DATA_VALIDATION_VALID_DIR
            )
        self.invalid_data_dir = os.path.join(
            self.data_validation_dir,
            constants.DATA_VALIDATION_INVALID_DIR
            )
        self.valid_train_file_path = os.path.join(
            self.valid_data_dir,
            constants.TRAIN_FILE_NAME
            )
        self.invalid_train_file_path = os.path.join(
            self.invalid_data_dir,
            constants.TRAIN_FILE_NAME
            )
        self.valid_test_file_path = os.path.join(
            self.valid_data_dir,
            constants.TEST_FILE_NAME
            )
        self.invalid_test_file_path = os.path.join(
            self.invalid_data_dir,
            constants.TEST_FILE_NAME
            )
        self.drift_report_file_path = os.path.join(
            self.data_validation_dir,
            constants.DATA_VALIDATION_DRIFT_REPORT_DIR,
            constants.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
            )
        self.schema_report_file_path = os.path.join(
            self.data_validation_dir,
            constants.DATA_VALIDATION_SCHEMA_REPORT_DIR,
            constants.DATA_VALIDATION_SCHEMA_REPORT_FILE_NAME
        )
        