import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging

import yaml
import json
import pandas as pd


def read_yaml_file(file_path: str):
    try:
        logging.info("Reading in yaml file")
        
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
        
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)
    
    
def read_csv_file(file_path: str):
    try:
        logging.info("Reading in csv file")
        
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)
    
def save_json(obj: object, file_path: str):
    try:
        with open(file_path, "w") as f:
            json.dump(obj, f)
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)