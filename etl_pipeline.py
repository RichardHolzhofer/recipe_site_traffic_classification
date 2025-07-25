import os
import sys
import json
from dotenv import load_dotenv
import pandas as pd
import pymongo
import certifi
import boto3
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.exception.exception import RecipeSiteTrafficException

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

ca = certifi.where()

class RecipeSiteTrafficETL():
    def __init__(self):
        try:
            self.file_path = os.path.join(os.getcwd(),"data","raw_s3","raw_data.csv")
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def extract_data_from_s3(self, bucket_name, key_id):
        try:
            
            
            logging.info("Connecting to AWS")
            
            s3 = boto3.client(
                service_name='s3',
                region_name = AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key = AWS_SECRET_ACCESS_KEY
            )
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            s3.download_file(Bucket=bucket_name, Key=key_id, Filename=self.file_path)
            
            logging.info(f"Downloaded file from S3 to {os.path.dirname(self.file_path)}")
            
        except Exception as e:
            logging.info("Connection failed to AWS, check your credentials")
            raise RecipeSiteTrafficException(e, sys)

    def csv_to_json_converter(self):
        try:
            logging.info("Converting csv to json format")
            
            df=pd.read_csv(self.file_path)
            df.reset_index(drop=True,inplace=True)
            df.columns = list(map(lambda x: x.lower(), df.columns))
            records = json.loads(df.to_json(orient='records'))
            return records
        
        except Exception as e:
            logging.info("Conversion failed")
            raise RecipeSiteTrafficException(e, sys)
        
    def load_data_to_mongodb(self,records,database, collection):
        try:
            logging.info("Loading records to MongoDB")
            
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            db = mongo_client[database]
            collection = db[collection]
            collection.insert_many(records)
            
            logging.info(f"Inserted {len(records)} records to MongoDB")
            
        except Exception as e:
            logging.info("Loading data to MongoDB failed")
            raise RecipeSiteTrafficException(e, sys)
        
if __name__ == "__main__":
    BUCKET = "recipe-site-traffic-classification"
    KEY_ID = "recipe-site-traffic-raw.csv"
    DATABASE = "recipe_db"
    COLLECTION = "traffic_data"
    
    etl_obj = RecipeSiteTrafficETL()
    etl_obj.extract_data_from_s3(
        bucket_name=BUCKET,
        key_id=KEY_ID
    )
    
    records = etl_obj.csv_to_json_converter()
    
    etl_obj.load_data_to_mongodb(
        records=records,
        database=DATABASE,
        collection=COLLECTION
    )
    