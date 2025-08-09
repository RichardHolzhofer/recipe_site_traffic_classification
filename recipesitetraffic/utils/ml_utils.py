import os
import sys
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.entity.artifact_entity import ClassificationMetricArtifact

import pandas as pd
import mlflow
from sklearn.metrics import precision_score, fbeta_score

from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
    
    

def hyperparameter_tuning(X_train, y_train, X_test, y_test, models, preprocessor, threshold, params={}):
    report = []
    best_model_name = None
    best_model = None
    best_params = None
    best_score = 0
    best_precision_score = 0
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        logging.info(f"Hyperparameter tuning started for {model_name}")
        param = params.get(model_name, {})
        
        pipe = ImbPipeline(steps=[("preprocessor", preprocessor), ('smote', SMOTE(random_state=42)),('model', model)])
        grid = GridSearchCV(pipe, param, cv=skf, n_jobs=-1, scoring='precision')
                    
                    
        grid.fit(X_train, y_train)
    
        with mlflow.start_run(run_name=f"Training pipeline: {model_name}"):
            y_train_pred = grid.best_estimator_.predict(X_train)
            y_test_pred = grid.best_estimator_.predict(X_test)
            
            train_metrics = get_scores(y_train, y_train_pred)
            test_metrics = get_scores(y_test, y_test_pred)
            
            test_scores = {
            "model_name": model_name,
            "test_precision_score":test_metrics.precision_score,
            "test_fbeta_score":test_metrics.fbeta_score
            }
            
            mlflow.log_metric("train_precision_score", train_metrics.precision_score)
            mlflow.log_metric("train_fbeta_score", train_metrics.fbeta_score)
            mlflow.log_metric("test_precision_score", test_metrics.precision_score)
            mlflow.log_metric("test_fbeta_score", test_metrics.fbeta_score)
        
            
        
        report.append(test_scores)
        
        if test_scores['test_fbeta_score'] > best_score:
            best_score = test_scores['test_fbeta_score']
            best_precision_score = test_scores['test_precision_score']
            best_model_name = model_name
            best_model = grid.best_estimator_
            best_params = grid.best_params_
                    
    if best_precision_score < threshold:
        raise ValueError(f"No model achieved the set precision threshold of {threshold}")
            
    results_df = pd.DataFrame(report).set_index("model_name").round(5)   
    print(f"Best precision score is {best_precision_score:.3f}, best fbeta score is: {best_score:.3f} for model: {best_model_name}")
    print(f"Best params for {best_model_name}: {best_params}")
        

    return results_df, best_model, best_model_name