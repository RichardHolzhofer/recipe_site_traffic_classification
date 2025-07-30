import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.entity.config_entity import DataTransformationConfig
from recipesitetraffic.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from recipesitetraffic.utils.main_utils import read_csv_file, save_numpy_array, save_object
from recipesitetraffic.constants.constants import TARGET_COLUMN

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import pandas as pd
import numpy as np
import joblib

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def create_preprocessor_object(self, train_df):
        try:
            
            logging.info("Creating preprocessor objects")
            cat_features = train_df.select_dtypes(include='object').columns.tolist()
            num_features = train_df.select_dtypes(include='number').columns.tolist()
            num_features.remove(TARGET_COLUMN)
            
            numerical_pipeline = ImbPipeline([("num_imputer", SimpleImputer(strategy='median')), ('pt', PowerTransformer())])
            categorical_pipeline = ImbPipeline([("cat_imputer", SimpleImputer(strategy='most_frequent')), (('ohe', OneHotEncoder(handle_unknown='ignore')))])
            
            ct = ColumnTransformer([('num_pipe', numerical_pipeline, num_features), ('cat_pipe', categorical_pipeline, cat_features)])
            
            smote_pipe = ImbPipeline(steps=[("ct", ct), ('smote', SMOTE(random_state=42))])
            tree_pipe = Pipeline(steps=[("ct", ct)])
            
            logging.info("Preprocessor with and without upsampling is created")
            
            return smote_pipe, tree_pipe
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
            
        
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            
            logging.info("Cleaning dataset has started")

            df.drop('recipe', axis=1, inplace=True)
            df.loc[df['category']=='Chicken Breast', 'category'] = 'Chicken'
            df['servings'] = df['servings'].str.strip(' as a snack')
            
            traffic_dict = {'High':1, np.nan:0}
            df['high_traffic'] = df['high_traffic'].map(traffic_dict)
            
            df.drop(df[(df.isna().sum(axis=1) >= 4)].index, inplace=True)
            df.reset_index(drop=True, inplace=True)

            if df.isna().any(axis=1).sum() <= len(df) * 0.05:
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                
            logging.info("Dataset has been cleaned")
                
            return df
            
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            
            logging.info("Data transformation has started")
            train_df = read_csv_file(self.data_validation_artifact.valid_train_file_path)
            test_df = read_csv_file(self.data_validation_artifact.valid_test_file_path)
            
            cleaned_train_df = self.clean_dataset(train_df)
            cleaned_test_df = self.clean_dataset(test_df)
            
            smote_pipe, tree_pipe = self.create_preprocessor_object(cleaned_train_df)
            
            logging.info("Splitting datasets into predictors and target feature")
            
            input_features_train_df = cleaned_train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = cleaned_train_df[TARGET_COLUMN]
            
            input_features_test_df = cleaned_test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = cleaned_test_df[TARGET_COLUMN]
            
            logging.info("Transforming train dataset without upsampling")
            transformed_train_df_features_normal = tree_pipe.fit_transform(input_features_train_df)
            
            logging.info("Transforming train dataset with upsampling")
            transformed_train_df_features_upsampled, transformed_train_df_target_feature_upsampled = smote_pipe.fit_resample(input_features_train_df, target_feature_train_df)
            
            logging.info("Transforming test dataset without upsampling")
            transformed_test_df_features = tree_pipe.transform(input_features_test_df)
            
            logging.info("Concatenating numpy array along Y axis")
            train_arr_normal = np.c_[transformed_train_df_features_normal, np.array(target_feature_train_df)]
            train_arr_upsampled = np.c_[transformed_train_df_features_upsampled, np.array(transformed_train_df_target_feature_upsampled)]
            
            test_arr = np.c_[transformed_test_df_features, np.array(target_feature_test_df)]
            
            logging.info("Setting up directories for preprocessed data artifacts")
            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.preprocessed_data_dir, exist_ok=True)
            
            logging.info("Saving transformed datasets")
            save_numpy_array(train_arr_normal, self.data_transformation_config.preprocessed_train_normal_file_path)
            save_numpy_array(train_arr_upsampled, self.data_transformation_config.preprocessed_train_upsampled_file_path)
            save_numpy_array(test_arr, self.data_transformation_config.preprocessed_test_file_path)
            
            logging.info("Setting up directories for data preprocessor artifacts")
            os.makedirs(self.data_transformation_config.preprocessor_object_dir, exist_ok=True)
            
            logging.info("Saving transformed preprocessors")
            save_object(tree_pipe, self.data_transformation_config.preprocessor_object_normal_file_path)
            save_object(smote_pipe, self.data_transformation_config.preprocessor_object_upsampler_file_path)
            
            logging.info("Data transformation has been completed successfully")
            return DataTransformationArtifact(
                preprocessor_object_normal_file_path=self.data_transformation_config.preprocessor_object_normal_file_path,
                preprocessor_object_upsampler_file_path=self.data_transformation_config.preprocessor_object_upsampler_file_path,
                preprocessed_train_normal_file_path= self.data_transformation_config.preprocessed_train_normal_file_path,
                preprocessed_train_upsampled_file_path=self.data_transformation_config.preprocessed_train_upsampled_file_path,
                preprocessed_test_file_path=self.data_transformation_config.preprocessed_test_file_path
                )
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        