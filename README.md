# Recipe Site Traffic Classification Project

The marketing department needs a reliable way to identify popular recipes to increase website traffic. This project solves that problem by developing a binary classification model that predicts whether a recipe will drive high or low traffic, meeting their requirement of 80% precision.

---

## 🚀 Project Status and Results

The project successfully developed a classification model that predicts high-traffic recipes with a **precision of ~ 92%**. This solution provides a valuable tool for content strategy, far exceeding the marketing department's 80% precision requirement.

**Key Findings:**
* **Recipe Categories are Key Predictors:** The recipe's category is a strong indicator of traffic. Recipes in the **Vegetable** category are a strong driver of high traffic.
* **Categories to Avoid:** The marketing team should be cautious when posting recipes from the **Beverages, Breakfast**, or **Chicken** categories, as they are less likely to result in high traffic.
* **Nutritional Information:** The model found no strong correlation between a recipe's nutritional content and its traffic performance.

The final model is currently deployed on the following URL: [https://recipesitetrafficclassification.streamlit.app](https://recipesitetrafficclassification.streamlit.app) where both single and batch predictions are available.

---

## 🛠️ Tech Stack and Project Structure

### Tech Stack
* **Machine Learning:** Python, Scikit-learn, XGBoost, CatBoost
* **MLOps:** DVC, Dagshub, MLflow
* **Data Validation:** Great Expectations, Evidently
* **Deployment:** Streamlit, FastAPI, Docker, AWS S3, AWS ECR, AWS EC2
* **CI/CD:** GitHub Actions, Ruff, Pytest
* **Databases:** MongoDB

### Folder Tree

├── .github/
│   └── workflows/
│       └── cicd.yaml
├── .streamlit/
│   └── secrets.toml.example
├── data/
│   ├── cleaned/
│   └── raw_s3/
├── data_schema/
│   └── schema.yaml
├── final_model/
│   └── final_model_with_preprocessor.joblib
├── notebooks/
│   ├── 1_EDA.ipynb
│   └── 2_Model_Training.ipynb
├── recipesitetraffic/
│   ├── components/
│   ├── constants/
│   ├── entity/
│   ├── exception/
│   ├── logging/
│   ├── pipeline/
│   └── utils/
├── templates/
│   ├── table.html
│   └── upload_form.html
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_data_validation.py
│   ├── test_data_transformation.py
│   └── test_model_trainer.py
├── .env.example
├── app.py
├── Dockerfile
├── etl_pipeline.py
├── main.py
├── requirements.txt
├── setup.py
├── streamlit_app.py
└── template.py

---

## ⚙️ Development and Pipelines

### ETL & Training Pipelines
* An ETL pipeline (`etl_pipeline.py`) extracts data from AWS S3, converts it to JSON format, and uploads it to MongoDB.
* The training pipeline consists of **Data Ingestion, Validation, Transformation, and Model Training**.
* **Data Validation** is performed using **Great Expectations** for schema checks and **Evidently** for data drift.
* **MLflow** is used for experiment tracking, with **Dagshub** serving as the MLflow server.
* The final model artifact, including the preprocessor, is saved and stored on **AWS S3**.

### Deployment & CI/CD
* The project includes a **Dockerized FastAPI application** for batch predictions, exposed on `0.0.0.0:8000`.
* A `cicd.yaml` file automates a CI/CD pipeline using **GitHub Actions**. This pipeline performs **code linting with Ruff**, runs **tests with Pytest**, and automates **deployment to AWS** via pushing the Docker image to **AWS ECR** and deploying it to an **AWS EC2** instance.

---

## 📧 Contact

If you have any questions, feel free to reach out to me at richard.holzhofer@gmail.com.

Thanks for reading!