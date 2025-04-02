import os
import pandas as pd
from sqlalchemy import create_engine
from prefect import task, flow
import time

# Variables de Entorno de la base de datos
db_user = "postgres"
db_password = "postgres"
db_host = "localhost"
db_port = "5432"
db_name = "sistema_electrico_raw"

engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Ruta a la carpeta con archivos JSON
folder_path = './data/'

def convertir_a_numerico(columna):
    # Intentar convertir a datetime para detectar si es una columna de fechas
    try:
        pd.to_datetime(columna)
        return columna  # Si no hay error, es una columna de fechas, devolverla sin cambios
    except (ValueError, TypeError):
        # Intentar convertir a numérico si no es una columna de fechas
        try:
            return pd.to_numeric(columna)
        except ValueError:
            return columna

@task
def list_files(folder_path):
    """Lista todos los archivos JSON en la carpeta especificada."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    return files

@task
def read_json_file(file_path):
    """Lee un archivo JSON y lo convierte en un DataFrame de pandas."""
    df = pd.read_json(file_path, lines=True)
    df = df.apply(convertir_a_numerico)
    return df

@task
def load_to_postgres(df, table_name):
    """Carga un DataFrame en una tabla PostgreSQL."""
    start_time = time.time()
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Tabla {table_name} cargada con éxito en {execution_time:.4f} segundos.')

@task
def get_table_name(file_name):
    """Obtiene el nombre de la tabla a partir del nombre del archivo."""
    return os.path.splitext(file_name)[0]

@task
def get_file_path(folder_path, file_name):
    """Obtiene la ruta completa del archivo."""
    return os.path.join(folder_path, file_name)

@flow
def flujo_carga():
    files = list_files(folder_path)
    for file_name in files:
        file_path = get_file_path(folder_path, file_name)
        df = read_json_file(file_path)
        table_name = get_table_name(file_name)
        load_to_postgres(df, table_name)

if __name__=="__main__":
    flujo_carga()
