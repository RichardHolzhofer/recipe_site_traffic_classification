# Recipe Site Traffic Classification Project

The marketing department selects a recipe each day to post on the homepage of the company's website. They have noticed that traffic to the rest of the website increases by as much as 40% if they pick a popular recipe. Since high traffic leads to more subscriptions to the company's services, they want to maximize traffic to their website.
Currently, they don't know how to determine whether a recipe will be popular, so they want a reliable solution that can select popular recipes at least 80% of the time.

The aim of this project is to support a company by developing a binary classification model for predicting whether a recipe will drive high or low traffic on their website.

## Project Status and Results

The project successfully developed a classification model that predicts high-traffic recipes with a precision of ~85 %. The solution meets the marketing department's requirement of a minimum 80% accuracy and provides a valuable tool for their content strategy.

**Key Findings:**
* **Recipe Categories are Key Predictors:** The type of recipe is a strong indicator of traffic. Some categories are far more likely to generate high traffic than others.
* **Categories to Prioritize:** Recipes in the **Vegetable** category are a strong driver of high traffic.
* **Categories to Avoid:** The marketing team should be cautious when posting recipes from the **Beverages, Breakfast**, or **Chicken** categories, as they are less likely to result in high traffic.
* **Nutritional Information:** The model did not find a strong correlation between a recipe's nutritional content (calories, carbohydrates, etc.) and its traffic performance. Other factors, like the recipe category, were more significant.predictors of traffic."

The final model is currently deployed on the following url: https://recipesitetrafficclassification.streamlit.app where single and batch predicitons are also available.

The EDA and the model training is documented in Jupyter notebooks under the notebooks/ folder.
(Datasets and models used in this stage are tracked with Dagshub remote DVC.)
Tested models:
 * LogisticRegression
 * KNeighborsClassifier
 * DecisionTreeClassifier
 * RandomForestClassifier
 * GradientBoostingClassifier
 * AdaBoostClassifier
 * XGBoostClassifier
 * CatBoostClassifier

## Developement and usage

The project uses custom Exception-handling and logging and uses a setup.py to be used as a package.

To be able to run the project, environmental variables must be set beforehand, refer to .env.example.


The ETL pipeline can be triggered by the following command after setting up the environmental variables:
```bash
python etl_pipeline.py

This extracts data from AWS S3, converts the csv to json format and uploads it to Mongo DB.

The training pipeline consists of 4 parts and artifacts are stored in timestamped folders locally, furthermore sent to AWS S3 at the end of the training pipeline.
* Data ingestion
* Data validation
* Data transformation
* Model training

In the data ingestion stage we extract the data from Mongo DB, do a light cleaning and extract the full dataset into our feature store as well as split the data to train and test set

In the data validation stage we check the data schema using the great_expectations library and check data drift with evidently between the train and test set. We save the schema and drift reports, as well as the datasets into the approriate folder based on the outcome of the reports.

In the data transformation stage we create a ColumnTransformer object which takes care of cleaning, scaling, imputing and encoding of the data. The preprocessor object is saved here with the transformed datasets.

In the model training stage we do hyperparameter tuning on the best models found based on our Jupyter notebook. Experiments are tracked with MLflow using Dagshub as our MLflow server. We pick the best performing model here and combine it with the preprocessor and save it as our final model.

Dockerfile is also available for the project which exposes a FastAPI application prepared for single and batch predicitons as well.

There is a cicd.yaml file also developed for CICD purposes. This is responsible for triggering Continuous Integration and Continuous Deployment/Delivery.
The Integration part contains code linting with Ruff and light testing with Pytest.
The Deployment part contains pushing our Dockerfile to AWS ECR.
The Delivery part contains pulling our image from AWS ECR and building and running the image on our AWS EC2 server.

├── data/
│   ├── cleaned/
│   └── raw_s3/
├── notebooks/
├── data_schema
│   ├── schema.yaml

├── recipesitetraffic/
│   ├── __init__.py
│   ├── components/
│       ├── __init__.py
│       ├── data_ingestion.py
│       ├── data_validation.py
│       ├── data_transformation.py
│       └── model_trainer.py
│   ├── constants/
│       ├── __init__.py
│       └── constants.py
│   ├── entity/
│       ├── __init__.py
│       ├── artifact_entity.py
│       └── config_entity.py
│   ├── exception/
│       ├── __init__.py
│       └── exception.py
│   ├── logging/
│       ├── __init__.py
│       └── logging.py
│   ├── pipeline/
│       ├── __init__.py
│       └── training_pipeline.py
│   └──  utils/
│       ├── __init__.py
│       ├── cloud.py
│       ├── estimator.py
│       ├── main_utils.py
│       ├── ml_utils.py
│       └── preprocessor.py
├── templates/
│       ├── table.html
│       └── upload_form.html
├── reports/
│   ├── figures/
│   └── final_report.pdf
├── tests/
│       ├── test_data_ingestion.py
│       ├── test_data_validation.py
│       ├── test_data_transformation.py
│       └──test_model_trainer.py
├── README.md
├── requirements.txt
└── .gitignore


For easier usage the final model with the preprocessor is committed to GitHub and is available under final_model/final_model_with_preprocessor.joblib


