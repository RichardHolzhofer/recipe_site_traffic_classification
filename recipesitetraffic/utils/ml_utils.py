import os
import sys
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.exception.exception import RecipeSiteTrafficException
import pandas as pd

def clean_for_drift_detection(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Replacing missing values in target column for drift detection, dropping lines with missing values")
        df = df.copy()
        df['high_traffic'] = df["high_traffic"].fillna("Low")
        df.dropna(inplace=True)
            
        return df
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)


