import os
import sys
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.exception.exception import RecipeSiteTrafficException
import pandas as pd

def clean_for_drift_detection(df):
    df['high_traffic'] = df["high_traffic"].fillna("Low")
    df.dropna(inplace=True)
        
    return df


