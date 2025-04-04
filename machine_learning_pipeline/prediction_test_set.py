import os
import pandas as pd
from sqlalchemy import create_engine

# Obtén las variables de entorno (o valores por defecto si no están definidas)
db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "postgres")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name_transformed = os.getenv("DB_NAME_TRANSFORMED", "sistema_electrico_features")

# Crea la cadena de conexión
connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name_transformed}"

# Crea el engine para conectarse a la base de datos
engine = create_engine(connection_string)

# Lee la tabla "sistema_electrico_features" en un DataFrame de pandas
df = pd.read_sql("SELECT * FROM sistema_electrico_features", con=engine)

# Toma aleatoriamente el 30% de las filas y reinicia el índice
df_test = df.sample(frac=0.3).reset_index(drop=True)

# Agrega la columna "batch" que asigne un número de lote cada 100 filas
df_test["batch"] = (df_test.index // 100) + 1

# Guarda el DataFrame resultante en la tabla "sistema_electrico_test_set"
# Crea la tabla si no existe, si existe se reemplaza
df_test.to_sql("sistema_electrico_test_set", engine, if_exists="replace", index=False)

print("Datos guardados correctamente en la tabla 'sistema_electrico_test_set' con la columna 'batch'.")