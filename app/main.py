from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_pipeline
from model.model import __version__ as model_version

app=FastAPI()

# Create a Request Model
class InputData(BaseModel):
    text:str

# Create output model
class PredictionOut(BaseModel):
    language:str

@app.get("/")
def home():
    return {"health check":"OK", "model_version":model_version}


#endpoints for prediction
@app.post("/predict/",response_model=PredictionOut)
def predict(payload:InputData):
    language=predict_pipeline(payload.text)
    return {"language":language}

