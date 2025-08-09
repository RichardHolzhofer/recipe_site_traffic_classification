import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging

import pandas as pd



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        
        logging.info("Cleaning data for prediction")
        df = df.copy()

        if 'recipe' in df.columns:
            df = df.drop('recipe', axis=1)

        df.loc[df['category'] == 'Chicken Breast', 'category'] = 'Chicken'

        if df['servings'].dtype == object:
            df['servings'] = df['servings'].str.replace(' as a snack', '', regex=False)

        df = df.drop(df[df.isna().sum(axis=1) >= 4].index)
        df = df.reset_index(drop=True)

        logging.info("Data is cleaned for prediction")
        return df
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)