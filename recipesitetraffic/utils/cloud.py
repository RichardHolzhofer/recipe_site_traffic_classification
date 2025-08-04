import os
import sys
import subprocess

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging

class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        try:
            logging.info("Syncing folder to S3")
            command = ["aws", "s3", "sync", folder, aws_bucket_url]
            result = subprocess.run(command, capture_output=True, text=True)
            
            return result
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def sync_folder_from_s3(self, folder, aws_bucket_url):
        try:
            logging.info("Syncing folder from S3")
            command = ["aws", "s3", "sync", aws_bucket_url, folder]
            result = subprocess.run(command, capture_output=True, text=True)
            
            return result
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)