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

The final model is currently deployed on the following url: https://recipesitetrafficclassification-2ixkfkfnw9lzyescvtpnxv.streamlit.app/ where single and batch predicitons are also available.

## Developement and usage

The ETL pipeline can be triggered by the following command after setting up the environmental variables:
```bash
python etl_pipeline.py

This extracts data from AWS S3, converts the csv to json format and uploads it to Mongo DB.

The training pipeline consists of 4 parts:
* Data ingestion
* Data validation
* Data transformation
* Model training



