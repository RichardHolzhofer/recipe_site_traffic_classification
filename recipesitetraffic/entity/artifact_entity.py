from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str
    
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    invalid_train_file_path: str
    valid_test_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str
    schema_report_file_path: str
    
@dataclass
class DataTransformationArtifact:
    preprocessor_object_normal_file_path: str
    preprocessor_object_upsampler_file_path: str
    preprocessed_train_normal_file_path: str
    preprocessed_train_upsampled_file_path: str
    preprocessed_test_file_path: str
    

