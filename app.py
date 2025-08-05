import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.pipeline.training_pipeline import TrainingPipeline
from recipesitetraffic.utils.main_utils import read_object
from recipesitetraffic.constants.constants import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME
from recipesitetraffic.utils.preprocessor import clean_data
from recipesitetraffic.constants.constants import FINAL_MODEL_FILE_PATH
from recipesitetraffic.utils.estimator import RecipeSiteTrafficBasicModel, RecipeSiteTrafficUpsamplerModel

import certifi
from dotenv import load_dotenv
import pymongo
import pandas as pd
from datetime import datetime

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca = certifi.where()

client = pymongo.MongoClient(MONGO_DB_URL, tlsCaFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]
templates = Jinja2Templates(directory="./templates")
last_prediction_table_html = ""

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=origins,
    allow_headers=origins
)

#@app.get("/", tags=["authentication"])
#async def index():
#    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
        return Response("Training is successful")
    
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)
    
    
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
def get_prediction(request: Request):
    if not last_prediction_table_html:
        return HTMLResponse("<h3>No prediction data available. Please upload first.</h3>")
    return templates.TemplateResponse("table.html", {"request": request, "table": last_prediction_table_html})
    
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile=File(...)):
    try:
        df = pd.read_csv(file.file)
        cleaned_df = clean_data(df)
        
        model = read_object(FINAL_MODEL_FILE_PATH)
        
        y_pred = model.predict(cleaned_df)
        
        cleaned_df['predicted_traffic'] = y_pred
        cleaned_df['predicted_traffic'] = cleaned_df['predicted_traffic'].map(lambda x: 'High' if x == 1 else 'Low')
        print(cleaned_df['predicted_traffic'])
        
        os.makedirs("predicted_data", exist_ok=True)
        cleaned_df.to_csv(f"predicted_data/predictions_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.csv", index=False)
        
        global last_prediction_table_html
        last_prediction_table_html = cleaned_df.to_html(classes='table table-striped', index=False, justify="right")
        
        return templates.TemplateResponse("table.html", {"request": request, "table": last_prediction_table_html})
            
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)
    
if __name__ == "__main__":
    app_run(
        app=app,
        host="0.0.0.0",
        port=8000
        )
 
    