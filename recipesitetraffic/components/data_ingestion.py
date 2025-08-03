import os
import sys
from dotenv import load_dotenv
from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.entity.config_entity import DataIngestionConfig
from recipesitetraffic.entity.artifact_entity import DataIngestionArtifact
from recipesitetraffic.constants.constants import TARGET_COLUMN

import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
                       
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def export_collection_as_dataframe(self):
        try:
            logging.info("Extracting data from MongoDB")
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.tolist():
                df.drop(columns=['_id'], axis=1, inplace=True)
            df.replace({"na":np.nan}, inplace=True)
            
            logging.info("Data has been successfully extracted from MongoDB")
            return df               
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
        finally:
            self.mongo_client.close()
        
    def export_data_to_feature_store(self, df: pd.DataFrame):
        try:
            logging.info("Exporting data to feature store")
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_path)
            
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(feature_store_path, index=False, header=True)
            logging.info("Export to feature store was successful")
            
            return df
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def split_data_into_train_and_test(self, df: pd.DataFrame):
        try:
            logging.info("Splitting data into train and test sets")
            df['high_traffic'] = df['high_traffic'].apply(lambda x: 1 if x == 'High' else 0)
            train_df, test_df = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=df[TARGET_COLUMN],
                shuffle=True
                )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_df.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            
            logging.info("Splitting was successful")
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started")
            df = self.export_collection_as_dataframe()
            df = self.export_data_to_feature_store(df)
            
            self.split_data_into_train_and_test(df)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info("Data ingestion pipeline is completed")
            return data_ingestion_artifact
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)