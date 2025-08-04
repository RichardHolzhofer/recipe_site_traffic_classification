import os
import sys

from recipesitetraffic.exception.exception import RecipeSiteTrafficException
from recipesitetraffic.logging.logger import logging
from recipesitetraffic.pipeline.training_pipeline import TrainingPipeline
from recipesitetraffic.utils.main_utils import read_object
from recipesitetraffic.constants.constants import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME

import certifi
from dotenv import load_dotenv
import pymongo
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=origins,
    allow_headers=origins
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
        return Response("Training is successful")
    
    except Exception as e:
        raise RecipeSiteTrafficException(e, sys)
    
    
if __name__ == "__main__":
    app_run(
        app=app,
        host="localhost",
        port=8000
        )