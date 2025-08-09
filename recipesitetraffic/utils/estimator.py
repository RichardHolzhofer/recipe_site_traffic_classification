import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging



class RecipeSiteTrafficModel:
    def __init__(self, cleaner_preprocessor, model):
        try:
            self.preprocessor = cleaner_preprocessor
            self.model = model
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
    
    def transform(self, X):
        try:
            X_transform = self.preprocessor.transform(X)
            return X_transform
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def predict(self, X):
        try:
            X_transform = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_transform)
            
            return y_pred
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)