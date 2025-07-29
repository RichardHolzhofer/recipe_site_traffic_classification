import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from recipesitetraffic.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from recipesitetraffic.constants.constants import SCHEMA_FILE_PATH
from recipesitetraffic.utils.main_utils import read_yaml_file, read_csv_file, save_json
from recipesitetraffic.utils.ml_utils import clean_for_drift_detection

import great_expectations as gx
import pandas as pd
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
        
    def run_data_schema_validation(self, dataframe, target_col, col_num, min_row, max_row):
        
        try:
            logging.info("Data schema check started")
            
            suite_name = "recipe_suite"
            datasource_name = "recipe_data_source"
            data_asset_name = "recipe_data_asset"
            batch_definition_name = "recipe_batch_definition"
            
            context = gx.get_context()
            
            datasource = context.data_sources.add_pandas(name=datasource_name)
            data_asset = datasource.add_dataframe_asset(name=data_asset_name)

            batch_definition = data_asset.add_batch_definition_whole_dataframe(name=batch_definition_name)
            batch = batch_definition.get_batch(batch_parameters={"dataframe": dataframe})
            
            schema = read_yaml_file(SCHEMA_FILE_PATH)['columns']
        
            suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
            
            suite.add_expectation(expectation=gx.expectations.ExpectTableColumnCountToEqual(value=col_num))
            suite.add_expectation(expectation=gx.expectations.ExpectTableRowCountToBeBetween(min_value=min_row, max_value=max_row))
            suite.add_expectation(expectation=gx.expectations.ExpectTableColumnsToMatchSet(column_set=set(schema.keys()), exact_match=True))
                
                    
            for k, v in schema.items():
                suite.add_expectation(expectation=gx.expectations.ExpectColumnValuesToBeOfType(column=k, type_=v))
            
                if k != target_col:
        
                    suite.add_expectation(expectation=gx.expectations.ExpectColumnProportionOfNonNullValuesToBeBetween(column=k, min_value=0.8, max_value=1))
                
            result = batch.validate(expect=suite)
            
            result_success = result.success
            
            dir_path = os.path.dirname(self.data_validation_config.schema_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            save_json(result.to_json_dict(), self.data_validation_config.schema_report_file_path)
            
            logging.info("Data schema check successfully finished")
            
            return result_success
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
    
    def run_data_drift_validation(self, train_df, test_df):
        
        try:
            logging.info("Data drift check started")
            
            train_drift_df = clean_for_drift_detection(train_df)
            test_drift_df = clean_for_drift_detection(test_df)
            
            cat_features = train_drift_df.select_dtypes(include='object').columns.tolist()
            num_features = train_drift_df.select_dtypes(include='number').columns.tolist()
            
            schema = DataDefinition(numerical_columns=num_features, categorical_columns=cat_features)
            
            eval_data_1 = Dataset.from_pandas(train_drift_df, data_definition=schema)
            
            eval_data_2 = Dataset.from_pandas(test_drift_df, data_definition=schema)
            
            report = Report([DataDriftPreset()])

            result = report.run(eval_data_1, eval_data_2).json()
            
            if report.run(eval_data_1, eval_data_2).dict()['metrics'][0]['value']['count'] == 0:
                result_success = True
            else:
                result_success = False
                
            dir_path = os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            save_json(result, self.data_validation_config.drift_report_file_path)
            
            logging.info("Data drift check successfully finished")
            
            return result_success
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
    
    def determine_validation_status(self, train_schema_result, test_schema_result, drift_result):
        try:
            return all([train_schema_result, test_schema_result, drift_result])
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
        
        
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            train_df = read_csv_file(train_file_path)
            test_df = read_csv_file(test_file_path)
            
            train_schema_result = self.run_data_schema_validation(train_df, 'high_traffic', 8, min_row=500, max_row=900)
            test_schema_result = self.run_data_schema_validation(test_df, 'high_traffic', 8, min_row=100, max_row=400)
            
            drift_result = self.run_data_drift_validation(train_df, test_df)
            
            status = self.determine_validation_status(train_schema_result, test_schema_result, drift_result)
            
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)
            
            
            return DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_ingestion_artifact.train_file_path,
                invalid_train_file_path = None,
                valid_test_file_path= self.data_ingestion_artifact.test_file_path,
                invalid_test_file_path = None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                schema_report_file_path=self.data_validation_config.schema_report_file_path
                )
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        