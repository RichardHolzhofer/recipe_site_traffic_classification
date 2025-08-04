import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging

from recipesitetraffic.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, DataValidationArtifact
from recipesitetraffic.entity.config_entity import ModelTrainerConfig
from recipesitetraffic.constants.constants import TARGET_COLUMN, FINAL_MODEL_FILE_PATH

from recipesitetraffic.utils.main_utils import read_object, read_numpy_array, save_object, read_csv_file
from recipesitetraffic.utils.ml_utils import get_scores, hyperparameter_tuning
from recipesitetraffic.utils.estimator import RecipeSiteTrafficBasicModel, RecipeSiteTrafficUpsamplerModel


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  GradientBoostingClassifier, RandomForestClassifier


import pandas as pd
import mlflow
import dagshub

dagshub.init(repo_owner='RichardHolzhofer', repo_name='recipe_site_traffic_classification', mlflow=True)

mlflow.set_experiment('Recipe Site Traffic')

class ModelTrainer:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def train_model(self, x_train_basic, y_train_basic, x_train_upsampled, y_train_upsampled, x_test, y_test):
        try:
            
            logging.info("Model training method has started")
            models={
                'LogisticRegression':LogisticRegression(random_state=42, verbose=1),
                'RandomForestClassifier':RandomForestClassifier(random_state=42, verbose=1),
                'GradientBoostingClassifier':GradientBoostingClassifier(random_state=42, verbose=1)
                }
            
            params = {
                "LogisticRegression": [
                    {
                        'penalty': ['l2'],
                        'solver': ['lbfgs', 'liblinear'],
                        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                    },
                    {
                        'penalty': ['l1'],
                        'solver': ['liblinear', 'saga'],
                        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    },
                    {
                        'penalty': ['elasticnet'],
                        'solver': ['saga'],
                        'C': [0.01, 0.1, 1.0, 10.0],
                        'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
                    }
                ],

                "GradientBoostingClassifier": {
                    'loss': ['log_loss'],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    #'subsample': [0.6, 0.8, 1.0],
                    'n_estimators': [50, 100, 150, 200],
                    'max_depth': [3, 4, 5, 7],
                    #'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    #'max_features': ['sqrt', 'log2', None],
                    'criterion': ['friedman_mse', 'squared_error']
                },

                "RandomForestClassifier": {
                    'n_estimators': [50, 100, 150, 200],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    #'min_samples_leaf': [1, 2, 4],
                    #'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    'class_weight': [None, 'balanced']
                }
                }
            
            results = []
            param_list = []
            
            for model in models.items():
                with mlflow.start_run(run_name=f"Training_pipeline: {model[0]}"):
                    score, param = hyperparameter_tuning(
                        X_train_basic=x_train_basic,
                        y_train_basic=y_train_basic,
                        X_train_upsampled=x_train_upsampled,
                        y_train_upsampled=y_train_upsampled,
                        X_test=x_test,
                        y_test=y_test,
                        model=model,
                        params=params
                        )
                    mlflow.log_param("model_type", model[0])
                    mlflow.log_metric("test_precision_score", score["test_precision_score"])
                    mlflow.log_metric("test_fbeta_score", score["test_fbeta_score"])
                    mlflow.log_params(param[model[0]])
                    results.append(score)
                    param_list.append(param)
                
                
            logging.info("Tuning for all models is finished")
            results_df = pd.DataFrame(results).set_index("model_name").round(3)
            filtered_df = results_df[results_df["test_precision_score"] >= self.model_trainer_config.expected_score]
            
            #results_df.to_csv("filtered_results.csv")
            if filtered_df.empty:
                raise ValueError(f"No model has reached the precision threshold of {self.model_trainer_config.expected_score}")
            
            sorted_df = filtered_df.sort_values(by="test_fbeta_score", ascending=False)
            
            best_model_name = filtered_df.index[0]
            logging.info(f"Best model is: {best_model_name}")
            
            best_model_score = sorted_df.iloc[0]["test_fbeta_score"]
            logging.info(f"Fbeta score is {best_model_score} for {best_model_name}")
            
            param_dict = {k: v for d in param_list for k, v in d.items()}
            best_model_params = param_dict[best_model_name]
            
            best_model_with_params_loaded = models[best_model_name].set_params(**best_model_params)
            
            logging.info("Model training method has finished")
            return best_model_name, best_model_with_params_loaded, results_df            
                    
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Model training has been initiated")
            
            
            logging.info("Loading preprocessor objects")
            basic_preprocessor = read_object(self.data_transformation_artifact.preprocessor_object_basic_file_path)
            upsampler_preprocessor = read_object(self.data_transformation_artifact.preprocessor_object_upsampler_file_path)
            
            logging.info("Loading transformed data")
            train_basic_array = read_numpy_array(self.data_transformation_artifact.preprocessed_train_basic_file_path)
            train_upsampled_array = read_numpy_array(self.data_transformation_artifact.preprocessed_train_upsampled_file_path)
            test_array = read_numpy_array(self.data_transformation_artifact.preprocessed_test_file_path)
            
            logging.info("Splitting transformed data to input and target features")
            x_train_basic, y_train_basic, x_train_upsampled, y_train_upsampled, x_test, y_test = (
                train_basic_array[:, :-1],
                train_basic_array[:, -1],
                train_upsampled_array[:, :-1],
                train_upsampled_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            
            best_model_name, best_model_with_params_loaded, results = self.train_model(
                x_train_basic=x_train_basic,
                y_train_basic=y_train_basic,
                x_train_upsampled=x_train_upsampled,
                y_train_upsampled=y_train_upsampled,
                x_test=x_test,
                y_test=y_test,
                
            )
            
            logging.info("Creating RecipeSiteTrafficModel with the best model found and appropriate preprocessor")
            if best_model_name == "LogisticRegression":
                model = RecipeSiteTrafficUpsamplerModel(
                    preprocessor=upsampler_preprocessor,
                    model=best_model_with_params_loaded
                )
            else:
                model = RecipeSiteTrafficBasicModel(
                    preprocessor=basic_preprocessor,
                    model=best_model_with_params_loaded
                )
                
            
            logging.info("Loading in validated data for inferencing with best model")
            train_validated = read_csv_file(self.data_validation_artifact.valid_train_file_path)
            test_validated = read_csv_file(self.data_validation_artifact.valid_test_file_path)
            
            logging.info("Splitting validated data to input and target features")
            x_train_val = train_validated.drop(TARGET_COLUMN, axis=1)
            y_train_val = train_validated[TARGET_COLUMN]
            
            x_test_val = test_validated.drop(TARGET_COLUMN, axis=1)
            y_test_val = test_validated[TARGET_COLUMN]
            
            logging.info("Fitting best model")
            model.fit(x_train_val, y_train_val)
            
            logging.info("Inferencing with the best model for train and test data")
            y_pred_train = model.predict(x_train_val)
            y_pred_test = model.predict(x_test_val)
            
            logging.info("Creating ClassificationMetricArtifacts for train and test data")
            """
            Artifacts may score slightly lower than `test_results` because they operate on raw (validated) data.
            To improve consistency and deployment readiness, we may include data cleaning steps inside the preprocessing pipeline as a future improvement.
            """
            
            train_artifact = get_scores(y_train_val, y_pred_train)
            test_artifact = get_scores(y_test_val, y_pred_test)
            
            
            logging.info("Creating directories for model training artifacts")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)
            os.makedirs(self.model_trainer_config.test_results_dir, exist_ok=True)
            
            
            logging.info("Saving best model and results artifacts")
            save_object(model, self.model_trainer_config.trained_model_file_path)
            results.to_csv(self.model_trainer_config.test_results_file_path)
            
            logging.info("Saving best model with preprocessor")
            save_object(model, FINAL_MODEL_FILE_PATH)
            
            with mlflow.start_run(run_name=f"Final model: {best_model_name}"):
                mlflow.log_metric("train_precision_score", train_artifact.precision_score)
                mlflow.log_metric("train_fbeta_score", train_artifact.fbeta_score)
                mlflow.log_metric("test_precision_score", test_artifact.precision_score)
                mlflow.log_metric("test_fbeta_score", test_artifact.fbeta_score)
                mlflow.log_params(model.model.get_params())
                mlflow.log_artifact(FINAL_MODEL_FILE_PATH, artifact_path="models")
            
            logging.info("Model training method finished, best model found.")
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_artifact,
                test_metric_artifact=test_artifact
            )
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        