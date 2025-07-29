import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from recipesitetraffic.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from recipesitetraffic.constants.constants import SCHEMA_FILE_PATH
from recipesitetraffic.utils.main_utils import read_yaml_file, read_csv_file, save_json

import great_expectations as gx
import pandas as pd

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
        
    def run_data_validation(self, dataframe, target_col, col_num, min_row, max_row, df_type='train'):
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
        
        return result
        
        
        
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            train_df = read_csv_file(train_file_path)
            test_df = read_csv_file(test_file_path)
            
            train_validation = self.run_data_validation(train_df, 'high_traffic', 8, min_row=500, max_row=900)
            
            print(train_validation)
            
            
            
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        