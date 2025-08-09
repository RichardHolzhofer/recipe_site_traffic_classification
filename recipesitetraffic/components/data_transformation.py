import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.entity.config_entity import DataTransformationConfig
from recipesitetraffic.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from recipesitetraffic.utils.main_utils import read_csv_file, save_object
from recipesitetraffic.utils.preprocessor import clean_data
from recipesitetraffic.constants.constants import TARGET_COLUMN

from sklearn.preprocessing import PowerTransformer, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline

import pandas as pd


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
    def create_transformer_preprocessor_object(self) -> object:
        try:
            
            logging.info("Creating preprocessor objects")
            
            
            cat_features = ['category', 'servings']
            num_features = ['calories', 'carbohydrate', 'sugar', 'protein']
            
            numerical_pipeline = ImbPipeline([("num_imputer", SimpleImputer(strategy='median')), ('pt', PowerTransformer())])
            categorical_pipeline = ImbPipeline([("cat_imputer", SimpleImputer(strategy='most_frequent')), (('ohe', OneHotEncoder(handle_unknown='ignore')))])
            
            ct = ColumnTransformer([('num_pipe', numerical_pipeline, num_features), ('cat_pipe', categorical_pipeline, cat_features)], remainder='passthrough')
            
            logging.info("Preprocessor with and without upsampling is created")
        
            return  ct
                
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
    
    def create_cleaner_preprocessor_object(self) -> object:
        try:
            ft = FunctionTransformer(clean_data)
            
            return ft
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        
        
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            cat_features = ['category', 'servings']
            num_features = ['calories', 'carbohydrate', 'sugar', 'protein']
            
            
            logging.info("Data transformation has started")
            train_df = read_csv_file(self.data_validation_artifact.valid_train_file_path)
            test_df = read_csv_file(self.data_validation_artifact.valid_test_file_path)
            
            cleaner_preprocessor = self.create_cleaner_preprocessor_object()
            
            logging.info("Transforming dataset - cleaning")
            cleaned_train_df = cleaner_preprocessor.fit_transform(train_df)
            cleaned_test_df = cleaner_preprocessor.transform(test_df)
            
            
            transformer_preprocessor = self.create_transformer_preprocessor_object()
            
            
            logging.info("Splitting cleaned datasets")
            cleaned_train_df_features = cleaned_train_df.drop(TARGET_COLUMN, axis=1)
            cleaned_train_df_target = cleaned_train_df[TARGET_COLUMN]
            
            cleaned_test_df_features = cleaned_test_df.drop(TARGET_COLUMN, axis=1)
            cleaned_test_df_target = cleaned_test_df[TARGET_COLUMN]
            
            logging.info("Transforming dataset")
            cleaned_transformed_train_arr = transformer_preprocessor.fit_transform(cleaned_train_df_features)
            cleaned_transformed_test_arr = transformer_preprocessor.transform(cleaned_test_df_features)
            
            
            cols = num_features + list(transformer_preprocessor.named_transformers_['cat_pipe'].named_steps['ohe'].get_feature_names_out(cat_features))
            
            cleaned_transformed_train_df = pd.DataFrame(cleaned_transformed_train_arr, columns=cols)
            cleaned_transformed_test_df = pd.DataFrame(cleaned_transformed_test_arr, columns=cols)
            
            final_transformed_train_df = pd.concat([cleaned_transformed_train_df, cleaned_train_df_target], axis=1)
            final_transformed_test_df = pd.concat([cleaned_transformed_test_df, cleaned_test_df_target], axis=1)
            
            
            logging.info("Setting up directories for preprocessed data artifacts")
            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.preprocessed_data_dir, exist_ok=True)
            
            logging.info("Saving transformed datasets")
            final_transformed_train_df.to_csv(self.data_transformation_config.preprocessed_train_file_path, index=False, header=True)
            final_transformed_test_df.to_csv(self.data_transformation_config.preprocessed_test_file_path, index=False, header=True)
            
            logging.info("Setting up directories for data preprocessor artifacts")
            os.makedirs(self.data_transformation_config.preprocessor_object_dir, exist_ok=True)
            
            logging.info("Saving transformed preprocessors")
            save_object(cleaner_preprocessor, self.data_transformation_config.preprocessor_cleaner_object_file_path)
            save_object(transformer_preprocessor, self.data_transformation_config.preprocessor_transformer_object_file_path)
            
            logging.info("Data transformation has been completed successfully")
            return DataTransformationArtifact(
                preprocessor_cleaner_object_file_path=self.data_transformation_config.preprocessor_cleaner_object_file_path,
                preprocessor_transformer_object_file_path=self.data_transformation_config.preprocessor_transformer_object_file_path,
                preprocessed_train_file_path= self.data_transformation_config.preprocessed_train_file_path,
                preprocessed_test_file_path=self.data_transformation_config.preprocessed_test_file_path
                )
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)
        