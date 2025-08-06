import os
import sys
import io

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging

from recipesitetraffic.utils.main_utils import read_object
from recipesitetraffic.utils.preprocessor import clean_data

from recipesitetraffic.constants.constants import FINAL_MODEL_FILE_PATH, TRAINING_PIPELINE_ARTIFACT_BUCKET_NAME, TARGET_COLUMN

from datetime import datetime
import boto3
import streamlit as st
import pandas as pd


AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]

BUCKET = "recipe-site-traffic-classification"
KEY_ID = "recipe-site-traffic-raw.csv"

st.set_page_config(page_title="Recipe Site Traffic Predictor", layout="centered")

st.title("🍽️ Recipe Site Traffic Predictor")
st.markdown("Upload a CSV file to predict recipe site traffic (High/Low).")


@st.cache_resource
def get_data_from_s3(bucket_name, key_id):
    try:
        
        s3 = boto3.client(
            service_name='s3',
            region_name = AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        )
        
        buffer = io.BytesIO()
        s3.download_fileobj(Bucket=bucket_name, Key=key_id, Fileobj=buffer)
        buffer.seek(0)
        
        return pd.read_csv(buffer)
         
    except Exception as e:
        logging.info("Connection failed to AWS, check your credentials")
        raise RecipeSiteTrafficException(e, sys)


@st.cache_resource
def load_best_model_with_preprocessor():
    
    try:
        
        model_dir = os.path.dirname(FINAL_MODEL_FILE_PATH)
    
        os.makedirs(model_dir, exist_ok=True)

        if not os.path.exists(FINAL_MODEL_FILE_PATH):
            logging.info(f"Model file not found at {FINAL_MODEL_FILE_PATH}.")
            st.warning("Prediction functionality is unavailable because the model could not be loaded.")
            
        return read_object(FINAL_MODEL_FILE_PATH)
    
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)




def upload_csv():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            
            df = pd.read_csv(uploaded_file)
            st.write("Original Data Preview:")
            st.dataframe(df.head())

            
            cleaned_df = clean_data(df.copy())

           
            with st.spinner("Predicting traffic..."):
                y_pred = model.predict(cleaned_df)
                
                if not isinstance(y_pred, pd.Series):
                    y_pred = pd.Series(y_pred)
                
                cleaned_df['predicted_traffic'] = y_pred.map({1: 'High', 0: 'Low'})
            
            st.success("Prediction complete!")
            st.write("Predicted Results:")
            st.dataframe(cleaned_df)

            st.download_button(
                label="Download Predictions as CSV",
                data=cleaned_df.to_csv(index=False).encode('utf-8'),
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        
        except Exception as e:
            raise RecipeSiteTrafficException(e, sys)


def slider_and_selectbox_setup(df):
    feature_dict = {}
    for col in df.columns:
        if col != TARGET_COLUMN:
            if df[col].dtype != 'O':
                feature_dict[col]={'type': 'numerical', 'min': df[col].min(), 'max': df[col].max()}
            else:
                feature_dict[col]={'type': 'categorical', 'options':sorted(list(df[col].unique()))}
                
    return feature_dict
        
    
    
    
#############################################################################




raw_df = get_data_from_s3(bucket_name=BUCKET, key_id=KEY_ID)
cleaned_train_df = clean_data(raw_df)
FEATURE_CONFIG = slider_and_selectbox_setup(cleaned_train_df)
model = load_best_model_with_preprocessor()



prediction_mode = st.radio(
        "Choose Prediction Mode:",
        ("Batch Prediction (CSV Upload)", "Single Instance Prediction")
    )


if prediction_mode == "Batch Prediction (CSV Upload)":
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        
        df = pd.read_csv(uploaded_file)
        st.write("Original Data Preview:")
        st.dataframe(df.head())
            
        with st.spinner("Predicting traffic..."):
            clean_df = clean_data(df)
            y_pred = model.predict(clean_df)
            
            if not isinstance(y_pred, pd.Series):
                y_pred = pd.Series(y_pred)
            
            clean_df['predicted_traffic'] = y_pred.map({1: 'High', 0: 'Low'})
        
        st.success("Prediction complete!")
        st.write("Predicted Results:")
        st.dataframe(clean_df)

        st.download_button(
            label="Download Predictions as CSV",
            data=clean_df.to_csv(index=False).encode('utf-8'),
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

elif prediction_mode == "Single Instance Prediction":
    st.subheader("Single Instance Prediction")
    
    input_data = {}
    for feature, config in FEATURE_CONFIG.items():
        if config['type'] == 'numerical':
            input_data[feature] = st.slider(
                f"Select {feature}",
                min_value=config['min'],
                max_value=config['max'],
                value=(config['min'] + config['max']) / 2, # Default to middle
                key=f"slider_{feature}"
            )
        elif config['type'] == 'categorical':
            input_data[feature] = st.selectbox(
                f"Select {feature}",
                options=config['options'],
                key=f"selectbox_{feature}"
            )
    
    if st.button("Predict Single Instance"):
        single_instance_df = pd.DataFrame([input_data])
        st.write("Input Data for Prediction:")
        st.dataframe(single_instance_df)
        prediction = model.predict(single_instance_df)[0]
        
        label = 'High' if prediction == 1 else 'Low'
        st.success(f"Predicted Traffic: **{label}**")