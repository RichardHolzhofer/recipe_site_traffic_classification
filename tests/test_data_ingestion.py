import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from recipesitetraffic.components.data_ingestion import DataIngestion
from recipesitetraffic.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from recipesitetraffic.constants.constants import TARGET_COLUMN



@pytest.fixture
def sample_config(tmp_path):
    # Create a mock TrainingPipelineConfig with overridden artifact_dir
    class TestTrainingPipelineConfig(TrainingPipelineConfig):
        def __init__(self):
            self.pipeline_name = "test_pipeline"
            self.artifact_name = "test_artifacts"
            self.artifact_dir = str(tmp_path / "artifacts")
            self.model_dir = str(tmp_path / "model")
            self.timestamp = "test_timestamp"

    training_pipeline_config = TestTrainingPipelineConfig()
    return DataIngestionConfig(training_pipeline_config)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "recipe": [
            "001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
            "011", "012", "013", "014", "015", "016", "017", "018", "019", "020"
        ],
        "calories": [
            "NA", 35.48, 914.28, 97.03, 27.05, 691.15, 183.94, 299.14, 538.52, 248.28,
            170.12, 155.8, 274.63, 25.23, 217.14, 316.45, 454.27, 1695.82, 1090.75, 127.55
        ],
        "carbohydrate": [
            "NA", 38.56, 42.68, 30.56, 1.85, 3.46, 47.95, 3.17, 3.78, 48.54,
            17.63, 8.27, 23.49, 11.51, 6.69, 2.65, 1.87, 0.1, 4.65, 27.55
        ],
        "sugar": [
            "NA", 0.66, 3.09, 38.63, 0.8, 1.65, 9.75, 0.4, 3.37, 3.99,
            4.1, 9.78, 1.56, 10.32, 10, 4.68, 2.95, 0.39, 0.69, 1.51
        ],
        "protein": [
            "NA", 0.92, 2.88, 0.02, 0.53, 53.93, 46.71, 32.4, 3.79, 113.85,
            0.91, 11.55, 2.57, 9.57, 15.17, 79.71, 61.07, 33.17, 3.49, 8.91
        ],
        "category": [
            "Pork", "Potato", "Breakfast", "Beverages", "Beverages", "One Dish Meal", "Chicken Breast", "Lunch/Snacks",
            "Pork", "Chicken", "Beverages", "Breakfast", "Potato", "Vegetable", "Meat", "Meat", "Meat", "Meat", "Meat", "Chicken"
        ],
        "servings": [
            6, 4, 1, 4, 4, 2, 4, 4, 6, 2,
            1, 6, 4, 4, 4, 6, 2, 1, 6, 2
        ],
        "high_traffic": [
            "High", "High", "NA", "High", "NA", "High", "NA", "NA", "High", "NA",
            "NA", "NA", "High", "High", "High", "High", "High", "High", "High", "NA"
        ]
    })


def test_export_collection_as_dataframe(sample_config, sample_dataframe):
    ingestion = DataIngestion(sample_config)

    with patch("pymongo.MongoClient") as mock_client:
        
        mock_collection = MagicMock()
        mock_collection.find.return_value = sample_dataframe.to_dict(orient="records")

        
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db

       
        df = ingestion.export_collection_as_dataframe()

        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == sample_dataframe.shape
        assert all(col in df.columns for col in sample_dataframe.columns)
        assert "_id" not in df.columns  # Since your method should drop it if present

def test_export_data_to_feature_store_basic(sample_config, sample_dataframe):
    ingestion = DataIngestion(sample_config)

    # Just check it runs and file is created
    df_returned = ingestion.export_data_to_feature_store(sample_dataframe)
    assert os.path.exists(sample_config.feature_store_file_path)
    assert isinstance(df_returned, pd.DataFrame)
        
def test_split_data_into_train_and_test_basic(sample_config, sample_dataframe):
    ingestion = DataIngestion(sample_config)

    # Run the method
    ingestion.split_data_into_train_and_test(sample_dataframe)

    # Check if train and test files exist
    assert os.path.exists(sample_config.training_file_path)
    assert os.path.exists(sample_config.testing_file_path)
 
def test_initiate_data_ingestion_basic(sample_config, sample_dataframe):
    ingestion = DataIngestion(sample_config)
    
    # Mock the methods used inside initiate_data_ingestion to isolate the test
    ingestion.export_collection_as_dataframe = lambda: sample_dataframe
    ingestion.export_data_to_feature_store = lambda df: df
    ingestion.split_data_into_train_and_test = lambda df: None
    
    artifact = ingestion.initiate_data_ingestion()
    
    assert artifact.train_file_path == sample_config.training_file_path
    assert artifact.test_file_path == sample_config.testing_file_path