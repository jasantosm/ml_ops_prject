import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

import psycopg2


# Variables de entorno para MLflow
os.environ["MLFLOW_TRACKING_URI"] = "postgresql+psycopg2://postgres:postgres@localhost/mlflow_db"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "masoud"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Strong#Pass#2022"

# Variables de entorno para la base de datos de predicciones
db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "postgres")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name_transformed = os.getenv("DB_NAME_TRANSFORMED", "sistema_electrico_features")

EXPERIMENT_NAME = "XGBoost_Sistema_Electrico"
METRIC_TO_SORT = "mse"  # Métrica utilizada para seleccionar el mejor modelo

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
    Date: str
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
    Recibe una lista de registros (cada uno con Date y 14 features),
    retorna el precio_bolsa_sistema_daily predicho para cada registro.
    Además, almacena tanto las features como la fecha de predicción y las predicciones en la tabla 'predictions'.
    """

    # Preparamos los datos para el modelo
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

    # Realizamos las predicciones
    predictions = model.predict(input_list)

    # Conectamos a la base de datos y creamos/insertamos en la tabla
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name_transformed,
        user=db_user,
        password=db_password
    )
    try:
        with conn.cursor() as cur:
            # Creamos la tabla si no existe
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_date TIMESTAMP,
                    generacion_sistema_daily FLOAT,
                    generacion_ideal_sistema_daily FLOAT,
                    perdidas_sistema_daily FLOAT,
                    volumen_util_energia_sistema_daily FLOAT,
                    emisiones_sistema_daily FLOAT,
                    aportes_energia_sistema_daily FLOAT,
                    aportes_energia_mediaHist_sistema_daily FLOAT,
                    capacidad_util_energia_sistema_daily FLOAT,
                    demanda_real_sistema_daily FLOAT,
                    exportaciones_sistema_daily FLOAT,
                    importaciones_sistema_daily FLOAT,
                    precio_escasez_sistema_daily FLOAT,
                    comsumo_combustible_daily FLOAT,
                    disponibilidad_real_daily FLOAT,
                    predicted_precio_bolsa_sistema_daily FLOAT
                );
            """)
            
            # Insertamos cada registro junto a su predicción
            for idx, pred in enumerate(predictions):
                row_data = data[idx]
                cur.execute("""
                    INSERT INTO predictions (
                        prediction_date,
                        generacion_sistema_daily,
                        generacion_ideal_sistema_daily,
                        perdidas_sistema_daily,
                        volumen_util_energia_sistema_daily,
                        emisiones_sistema_daily,
                        aportes_energia_sistema_daily,
                        aportes_energia_mediaHist_sistema_daily,
                        capacidad_util_energia_sistema_daily,
                        demanda_real_sistema_daily,
                        exportaciones_sistema_daily,
                        importaciones_sistema_daily,
                        precio_escasez_sistema_daily,
                        comsumo_combustible_daily,
                        disponibilidad_real_daily,
                        predicted_precio_bolsa_sistema_daily
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    row_data.Date,
                    row_data.generacion_sistema_daily,
                    row_data.generacion_ideal_sistema_daily,
                    row_data.perdidas_sistema_daily,
                    row_data.volumen_util_energia_sistema_daily,
                    row_data.emisiones_sistema_daily,
                    row_data.aportes_energia_sistema_daily,
                    row_data.aportes_energia_mediaHist_sistema_daily,
                    row_data.capacidad_util_energia_sistema_daily,
                    row_data.demanda_real_sistema_daily,
                    row_data.exportaciones_sistema_daily,
                    row_data.importaciones_sistema_daily,
                    row_data.precio_escasez_sistema_daily,
                    row_data.comsumo_combustible_daily,
                    row_data.disponibilidad_real_daily,
                    float(pred)
                ))
        # Confirmamos los cambios
        conn.commit()
    finally:
        conn.close()

    # Preparamos la respuesta para retornar
    response = [
        PredictionResponse(precio_bolsa_sistema_daily=float(pred))
        for pred in predictions
    ]
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)