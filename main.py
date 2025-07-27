import os
import sys
from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.components.data_ingestion import DataIngestion
from recipesitetraffic.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig




if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        print(data_ingestion_artifact)
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)


