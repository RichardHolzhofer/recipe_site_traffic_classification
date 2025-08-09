import os
import sys
from dotenv import load_dotenv

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging

from recipesitetraffic.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, DataValidationArtifact
from recipesitetraffic.entity.config_entity import ModelTrainerConfig
from recipesitetraffic.constants.constants import TARGET_COLUMN, FINAL_MODEL_FILE_PATH

from recipesitetraffic.utils.main_utils import read_object, save_object, read_csv_file, save_json
from recipesitetraffic.utils.ml_utils import get_scores, hyperparameter_tuning
from recipesitetraffic.utils.estimator import RecipeSiteTrafficModel


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


import pandas as pd
import mlflow

load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.set_experiment('Recipe Site Traffic')

class ModelTrainer:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def clean_data_for_model_training(self, cleaner_preprocessor, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            logging.info("Cleaning dataset")
            
            cleaned_train_df = cleaner_preprocessor.fit_transform(train_df)
            cleaned_test_df = cleaner_preprocessor.transform(test_df)
            
            cleaned_train_features = cleaned_train_df.drop(TARGET_COLUMN, axis=1)
            cleaned_train_target = cleaned_train_df[TARGET_COLUMN]
            
            cleaned_test_features = cleaned_test_df.drop(TARGET_COLUMN, axis=1)
            cleaned_test_target = cleaned_test_df[TARGET_COLUMN]
            
            logging.info("Cleaning finished")
            
            return cleaned_train_features, cleaned_train_target, cleaned_test_features, cleaned_test_target
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def train_model(self, X_train, y_train, X_test, y_test, transformer_preprocessor):
        try:
            
            logging.info("Model training method has started")
            
            models = {
                'LogisticRegression':LogisticRegression(random_state=42, verbose=1),
                'DecisionTreeClassifier':DecisionTreeClassifier(random_state=42),
                'GradientBoostingClassifier':GradientBoostingClassifier(random_state=42, verbose=1)
                }
            
            params = {
                "LogisticRegression": [
                    {
                        'model__penalty': ['l2'],
                        'model__solver': ['lbfgs', 'liblinear'],
                        'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                    },
                    {
                        'model__penalty': ['l1'],
                        'model__solver': ['liblinear', 'saga'],
                        'model__C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    },
                    {
                        'model__penalty': ['elasticnet'],
                        'model__solver': ['saga'],
                        'model__C': [0.01, 0.1, 1.0, 10.0],
                        'model__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
                    }
                ],

                "GradientBoostingClassifier": {
                    'model__loss': ['log_loss'],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    #'model__subsample': [0.6, 0.8, 1.0],
                    'model__n_estimators': [50, 100, 150, 200],
                    'model__max_depth': [3, 4, 5, 7],
                    #'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    #'model__max_features': ['sqrt', 'log2', None],
                    'model__criterion': ['friedman_mse', 'squared_error']
                },

                "DecisionTreeClassifier": {
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__max_depth': [3, 5, 7, 10, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2', None],
                'model__class_weight': [None, 'balanced']
                }
                }
            
            results_df, best_model, best_model_name = hyperparameter_tuning(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                preprocessor=transformer_preprocessor,
                params=params,
                threshold=self.model_trainer_config.expected_score
            )
            
            return results_df, best_model, best_model_name
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)        
     

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            
            logging.info("Loading validated dataset")
            train_df = read_csv_file(self.data_validation_artifact.valid_train_file_path)
            test_df = read_csv_file(self.data_validation_artifact.valid_test_file_path)
            
            logging.info("Loading cleaner preprocessor")
            cleaner_preprocessor = read_object(self.data_transformation_artifact.preprocessor_cleaner_object_file_path)
            
            logging.info("Cleaning validated dataset")
            X_train, y_train, X_test, y_test = self.clean_data_for_model_training(
                cleaner_preprocessor=cleaner_preprocessor,
                train_df=train_df,
                test_df=test_df
                )
            
            logging.info("Loading transformer preprocessor")
            transformer_preprocessor = read_object(self.data_transformation_artifact.preprocessor_transformer_object_file_path)
            
            
            results_df, best_model, best_model_name = self.train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                transformer_preprocessor=transformer_preprocessor
            )
            
            #results_df.to_csv("results_df.csv")
            
            logging.info("Creating directories for model training artifacts")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)
            os.makedirs(self.model_trainer_config.test_results_dir, exist_ok=True)
            
           
            logging.info("Building RecipeSiteTrafficModel for inferencing")
            model = RecipeSiteTrafficModel(
                cleaner_preprocessor=cleaner_preprocessor,
                model=best_model
            )
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_artifact = get_scores(y_train, y_pred_train)
            test_artifact = get_scores(y_test, y_pred_test)
            
            
            test_artifact_dict = {"test_precision_score" :test_artifact.precision_score,
                                   "test_fbeta_score": test_artifact.fbeta_score
                                   }
            
            
            logging.info("Saving best model")
            save_object(model, FINAL_MODEL_FILE_PATH)
            
            logging.info("Saving best model with artifacts")
            save_object(best_model, self.model_trainer_config.trained_model_file_path)
            save_json(test_artifact_dict, self.model_trainer_config.test_results_file_path)
            
            logging.info("Logging metrics with mlflow")
            with mlflow.start_run(run_name=f"Final model: {best_model_name}"):
                mlflow.log_metric("train_precision_score", train_artifact.precision_score)
                mlflow.log_metric("train_fbeta_score", train_artifact.fbeta_score)
                mlflow.log_metric("test_precision_score", test_artifact.precision_score)
                mlflow.log_metric("test_fbeta_score", test_artifact.fbeta_score)
                mlflow.log_params(best_model.get_params())
                mlflow.log_artifact(FINAL_MODEL_FILE_PATH, artifact_path="models")
            
            logging.info("Model training method finished, best model found.")
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_artifact,
                test_metric_artifact=test_artifact
            )
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        