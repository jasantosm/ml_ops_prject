import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient



os.environ["MLFLOW_TRACKING_URI"] = "postgresql+psycopg2://postgres:postgres@localhost/mlflow_db"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "masoud"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Strong#Pass#2022"


EXPERIMENT_NAME = "XGBoost_Sistema_Electrico"
METRIC_TO_SORT = "mse"  # or "rmse", "r2", etc.

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=[f"metrics.{METRIC_TO_SORT} ASC"],
    max_results=1
)

best_run_id = runs[0].info.run_id
print(f"Best run_id: {best_run_id}, based on lowest {METRIC_TO_SORT}.")


MODEL_URI = f"runs:/{best_run_id}/xgboost_model"

model = mlflow.xgboost.load_model(MODEL_URI)

class InputData(BaseModel):
    generacion_sistema_daily: float
    generacion_ideal_sistema_daily: float
    perdidas_sistema_daily: float
    volumen_util_energia_sistema_daily: float
    emisiones_sistema_daily: float
    aportes_energia_sistema_daily: float
    aportes_energia_mediaHist_sistema_daily: float
    capacidad_util_energia_sistema_daily: float
    demanda_real_sistema_daily: float
    exportaciones_sistema_daily: float
    importaciones_sistema_daily: float
    precio_escasez_sistema_daily: float
    comsumo_combustible_daily: float
    disponibilidad_real_daily: float


class PredictionResponse(BaseModel):
    precio_bolsa_sistema_daily: float


app = FastAPI(
    title="Sistema Electrico API Predicción",
    description="API para predecir el precio bolsa a partir de las variables del sistema electrico Colombiano"
)


@app.post("/predict", response_model=List[PredictionResponse])
def predict(data: List[InputData]):
    """
    Receive a list of records (each with 14 features),
    return predicted `precio_bolsa_sistema_daily` for each.
    """
    # Convert input objects to the structure expected by the model
    input_list = []
    for row in data:
        input_list.append([
            row.generacion_sistema_daily,
            row.generacion_ideal_sistema_daily,
            row.perdidas_sistema_daily,
            row.volumen_util_energia_sistema_daily,
            row.emisiones_sistema_daily,
            row.aportes_energia_sistema_daily,
            row.aportes_energia_mediaHist_sistema_daily,
            row.capacidad_util_energia_sistema_daily,
            row.demanda_real_sistema_daily,
            row.exportaciones_sistema_daily,
            row.importaciones_sistema_daily,
            row.precio_escasez_sistema_daily,
            row.comsumo_combustible_daily,
            row.disponibilidad_real_daily
        ])

    predictions = model.predict(input_list)

    response = [
        PredictionResponse(precio_bolsa_sistema_daily=float(pred))
        for pred in predictions
    ]
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)