import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from recipesitetraffic.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from recipesitetraffic.constants.constants import SCHEMA_FILE_PATH
from recipesitetraffic.utils.main_utils import read_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            pass
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        