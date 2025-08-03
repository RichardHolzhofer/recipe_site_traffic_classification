import os
import sys
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.entity.artifact_entity import ClassificationMetricArtifact
from recipesitetraffic.utils.main_utils import read_object, read_numpy_array

import pandas as pd
from sklearn.metrics import precision_score, fbeta_score, make_scorer

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def clean_for_drift_detection(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Replacing missing values in target column for drift detection, dropping lines with missing values")
        df = df.copy()
        df['high_traffic'] = df["high_traffic"].fillna("Low")
        df.dropna(inplace=True)
            
        return df
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)


def get_scores(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        logging.info("Creating ClassificationMetricArtifact")
        return ClassificationMetricArtifact(
            precision_score=precision_score(y_true, y_pred),
            fbeta_score=fbeta_score(y_true, y_pred, beta=0.5)
            )
            
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)
    
    

def hyperparameter_tuning(X_train_basic, y_train_basic, X_train_upsampled, y_train_upsampled, X_test, y_test, model, params={}):
    
    logging.info(f"Hyperparameter tuning started for {model[0]}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fbeta_scorer = make_scorer(fbeta_score, beta=0.5)
    
    model_name, model = model
        
    param = params.get(model_name, {})

    if model_name == 'LogisticRegression':
    
        grid = GridSearchCV(model, param, cv=skf, n_jobs=-1, scoring=fbeta_scorer)
        grid.fit(X_train_upsampled, y_train_upsampled)
        
        logging.info("Inferencing for test data")
        y_train_pred = grid.best_estimator_.predict(X_train_upsampled)
        y_test_pred = grid.best_estimator_.predict(X_test)
        
        logging.info("Inferencing for test data")
        #train_metrics = get_scores(y_train_upsampled, y_train_pred)
        #test_metrics = get_scores(y_test, y_test_pred)
    
    else:
        grid = GridSearchCV(model, param, cv=skf, n_jobs=-1, scoring=fbeta_scorer)
        grid.fit(X_train_basic, y_train_basic)
        
        logging.info("Inferencing for test data")
        y_train_pred = grid.best_estimator_.predict(X_train_basic)
        y_test_pred = grid.best_estimator_.predict(X_test)

        #train_metrics = get_scores(y_train_basic, y_train_pred)
    test_metrics = get_scores(y_test, y_test_pred)

    scores = {
    "model_name": model_name,
    "test_precision_score":test_metrics.precision_score,
    "test_fbeta_score":test_metrics.fbeta_score
    }

    model_params = {model_name:grid.best_params_}
    print(f"Best precision score is {test_metrics.precision_score:.3f} for model: {model_name}")
    print(f"Best f-beta (0.5) score is {test_metrics.fbeta_score:.3f} for model: {model_name}")
    
    
    return scores, model_params