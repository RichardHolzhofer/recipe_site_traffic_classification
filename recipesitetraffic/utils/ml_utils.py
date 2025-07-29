import os
import sys
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.utils.main_utils import read_yaml_file
from recipesitetraffic.constants.constants import SCHEMA_FILE_PATH
import yaml
import great_expectations as gx




def add_expectations(suite_name: str, target_col: str, col_num: int, min_row: int, max_row: int):
    try:
        logging.info("Reading in data schema")
        context = gx.get_context()
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
        
        schema = read_yaml_file(SCHEMA_FILE_PATH)['columns']
            
        logging.info("Adding schema expectations to ExpectationSuite")
            
        suite.add_expectation(expectation=gx.expectations.ExpectTableColumnCountToEqual(value=col_num))
        suite.add_expectation(expectation=gx.expectations.ExpectTableRowCountToBeBetween(min_value=min_row, max_value=max_row))
        suite.add_expectation(expectation=gx.expectations.ExpectTableColumnsToMatchSet(column_set=set(schema.keys()), exact_match=True))
        
        logging.info("Adding type and completeness expectations to ExpectationSuite")
        for k, v in schema.items():
            suite.add_expectation(expectation=gx.expectations.ExpectColumnValuesToBeOfType(column=k, type_=v))
            
            if k != target_col:
                suite.add_expectation(expectation=gx.expectations.ExpectColumnProportionOfNonNullValuesToBeBetween(column=k, min_value=0.8, max_value=1))
                
        logging.info("All expectations have been successfully added to ExpectationSuite")
    
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)