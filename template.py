import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(message)s]"
)

project_name = 'recipesitetraffic'

list_of_files = [
    "notebooks/1_EDA.ipynb",
    "notebooks/2_Model_Training.ipynb",
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/monitor_drift.py",
    "tests/test_data_ingestion.py",
    "tests/test_data_validation.py",
    "tests/test_data_transformation.py",
    "tests/test_model_trainer.py",
    "tests/test_monitor_drift.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/constants/constants.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception.py",
    f"{project_name}/logging/__init__.py",
    f"{project_name}/logging/logger.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    f"{project_name}/utils/ml_utils.py",
    "data/.gitkeep",
    ".github/workflows/.gitkeep",
    ".gitignore",
    ".env",
    "app.py",
    "Dockerfile",
    "main.py",
    "README.md",
    "requirements.txt",
    "setup.py" 
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory {file_dir} for file {file_name}")
        
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        
        with open(file_path, "w") as file:
            pass
            logging.info(f"Creating empty file: {file_path}")
                
    else:
        logging.info(f"{file_name} already exists")
