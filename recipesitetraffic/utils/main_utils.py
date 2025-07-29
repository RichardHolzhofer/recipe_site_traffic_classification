import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging

import yaml


def read_yaml_file(file_path: str):
    try:
        logging.info("Reading in yaml file")
        
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
        
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)