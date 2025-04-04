import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# ML / XGBoost
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

# MLflow
import mlflow
import mlflow.xgboost  # Para loguear modelos de XGBoost
from mlflow.exceptions import MlflowException

# Prefect
from prefect import flow, task


# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE VARIABLES DE ENTORNO (MLFLOW Y BASE DE DATOS)
# -----------------------------------------------------------------------------
# Variables de Entorno de la base de datos
db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "postgres")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name_transformed = os.getenv("DB_NAME_TRANSFORMED", "sistema_electrico_features")

# MLflow con Postgres + MinIO
os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://postgres:postgres@localhost/mlflow_db'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://127.0.0.1:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'masoud'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'Strong#Pass#2022'

experiment_name = "XGBoost_Sistema_Electrico"

# Crear el experimento en MLflow si no existe
try:
    mlflow.create_experiment(
        experiment_name, 
        artifact_location="s3://mlflow"  # carpeta/bucket en MinIO
    )
except MlflowException as e:
    print(f"Experiment may already exist: {e}")

mlflow.set_experiment(experiment_name)


# -----------------------------------------------------------------------------
# 2. TAREAS DE PREFECT
# -----------------------------------------------------------------------------
@task
def load_data_from_postgres() -> pd.DataFrame:
    """
    Conecta a la base de datos Postgres y carga los datos en un DataFrame.
    Ajusta la consulta y nombre de la tabla a tu entorno.
    """
    # Conexión usando las variables de entorno para la base 'db_name_transformed'
    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name_transformed}"
    )
    
    # Ajusta la tabla y la consulta a tu caso
    query = "SELECT * FROM sistema_electrico_features;"
    df = pd.read_sql(query, engine)
    return df


@task
def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.3, random_state: int = 42):
    """
    Separa las features de la variable objetivo y luego realiza el train/test split.
    """
    X = df.drop([target_column, 'Date'], axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


@task
def hyperparameter_search(X_train, y_train):
    """
    Realiza la búsqueda de hiperparámetros utilizando GridSearchCV y KFold.
    Devuelve los mejores parámetros y el mejor modelo.
    """
    xgb_reg = XGBRegressor(objective='reg:squarederror')
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=kfold,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_, grid_search.best_estimator_


@task
def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo con MSE, RMSE y R^2. 
    Retorna un diccionario con las métricas.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


# -----------------------------------------------------------------------------
# 3. FLUJO PRINCIPAL DE PREFECT
# -----------------------------------------------------------------------------
@flow(name="XGBoost Model Training Pipeline")
def main_flow():
    """
    Orquesta el pipeline completo:
      - Carga datos de Postgres
      - Separa en train/test
      - Búsqueda y entrenamiento de hiperparámetros
      - Evaluación
      - Registro en MLflow
    """
    # 1. Carga de datos
    df = load_data_from_postgres()
    
    # 2. Separar características y variable objetivo, y split de train/test
    # Ajusta el nombre de la columna objetivo según tu tabla (ej: "precio_bolsa_sistema_daily")
    X_train, X_test, y_train, y_test = split_dataset(df, target_column="precio_bolsa_sistema_daily")
    
    # 3. Búsqueda de hiperparámetros
    best_params, best_estimator = hyperparameter_search(X_train, y_train)
    
    # 4. Entrenamiento final y logging en MLflow
    with mlflow.start_run(run_name="XGBoost_Prefect_Train"):
        # Log de hiperparámetros
        mlflow.log_params(best_params)
        
        # Entrenar el mejor modelo
        best_estimator.fit(X_train, y_train)
        
        # 5. Evaluación
        metrics = evaluate_model(best_estimator, X_test, y_test)
        
        # Log de métricas
        mlflow.log_metric("mse", metrics["mse"])
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("r2", metrics["r2"])
        
        # 6. Loguear el modelo en MLflow
        mlflow.xgboost.log_model(best_estimator, "xgboost_model")

        # Mensajes en consola (opcional)
        print("Mejores hiperparámetros:", best_params)
        print("MSE:", metrics["mse"])
        print("RMSE:", metrics["rmse"])
        print("R^2:", metrics["r2"])


# -----------------------------------------------------------------------------
# 4. EJECUCIÓN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main_flow()