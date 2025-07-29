import os
import sys
from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.components.data_ingestion import DataIngestion
from recipesitetraffic.components.data_validation import DataValidation
from recipesitetraffic.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig




if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact,data_validation_config)
        data_validation.initiate_data_validation()
        
        
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)


