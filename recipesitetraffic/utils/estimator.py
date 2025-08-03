import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging


class RecipeSiteTrafficUpsamplerModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
    
    def fit(self, X, y):
        X_transformed, y_transformed = self.preprocessor.fit_resample(X, y)
        self.model.fit(X_transformed, y_transformed)
        return self
        
    def predict(self, X):
        try:
            X_transform = self.preprocessor.named_steps['ct'].transform(X)
            y_pred = self.model.predict(X_transform)
            
            return y_pred
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)



class RecipeSiteTrafficBasicModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
    
    def fit(self, X, y):
        X_transformed, y_transformed = self.preprocessor.fit_transform(X, y)
        self.model.fit(X_transformed, y_transformed)
        return self
        
    def predict(self, X):
        try:
            X_transform = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_transform)
            
            return y_pred
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)