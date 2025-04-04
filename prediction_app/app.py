import os
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(page_title="Predictor", layout="wide")

# 1. Configuración de la conexión a la base de datos
db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "postgres")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name_transformed = os.getenv("DB_NAME_TRANSFORMED", "sistema_electrico_features")

connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name_transformed}"
engine = create_engine(connection_string)

# 2. Encabezado de la app
st.title("Predicciones de Sistema Eléctrico")

# 3. Obtener y mostrar el listado de batches disponibles
query_batches = """
    SELECT DISTINCT batch
    FROM sistema_electrico_test_set
    ORDER BY batch
"""
df_batches = pd.read_sql(query_batches, engine)

if df_batches.empty:
    st.error("No hay datos en la tabla 'sistema_electrico_test_set'.")
else:
    batch_list = df_batches["batch"].tolist()
    selected_batch = st.selectbox("Selecciona un batch:", batch_list)

    # 4. Al seleccionar un batch, generamos la consulta para obtener las columnas necesarias
    #    Mantenemos las columnas requeridas para la predicción.
    columns_required = [
        '"Date"',
        "generacion_sistema_daily",
        "generacion_ideal_sistema_daily",
        "perdidas_sistema_daily",
        "volumen_util_energia_sistema_daily",
        "emisiones_sistema_daily",
        "aportes_energia_sistema_daily",
        '"aportes_energia_mediaHist_sistema_daily"',
        "capacidad_util_energia_sistema_daily",
        "demanda_real_sistema_daily",
        "exportaciones_sistema_daily",
        "importaciones_sistema_daily",
        "precio_escasez_sistema_daily",
        "comsumo_combustible_daily",
        "disponibilidad_real_daily"
    ]

    query_data = f"""
        SELECT {', '.join(columns_required)}
        FROM sistema_electrico_test_set
        WHERE batch = {selected_batch}
    """
    df_data = pd.read_sql(query_data, engine)

    if df_data.empty:
        st.warning("No hay registros para el batch seleccionado.")
    else:
        st.write(f"Datos para batch: {selected_batch}")
        st.dataframe(df_data)

        # 5. Enviar datos al endpoint de FastAPI y recibir las predicciones
        if st.button("Generar Predicciones"):
            try:

                # ==============================
                #  EVITAR EL ERROR DE TIMESTAMP
                # ==============================
                # Convertimos la columna "Date" a string (ISO8601, por ejemplo)
                df_data["Date"] = pd.to_datetime(df_data["Date"]).dt.strftime("%Y-%m-%d")

                # Convertimos el dataframe a diccionario para enviarlo como JSON
                data_to_predict = df_data.to_dict(orient="records")

                # Petición POST al servicio de FastAPI (http://localhost:8000/predict)
                response = requests.post(
                    "http://localhost:8000/predict",
                    json=data_to_predict
                )

                if response.status_code == 200:
                    # Parseamos la respuesta (lista de diccionarios, uno por fila)
                    predictions = response.json()

                    # Agregamos la columna de predicciones al df_data
                    # Suponiendo que 'predictions' y df_data tienen la misma longitud,
                    # y que cada elemento de 'predictions' es {'precio_bolsa_sistema_daily': valor}
                    df_data["precio_bolsa_sistema_daily"] = [
                        item["precio_bolsa_sistema_daily"] for item in predictions
                    ]

                    # Convertimos Date a tipo fecha/hora para poder usarlo como índice
                    df_data["Date"] = pd.to_datetime(df_data["Date"])
                    df_data.set_index("Date", inplace=True)

                    st.success("Predicciones generadas correctamente.")

                    # 6. Mostramos el resultado en un gráfico de línea
                    st.line_chart(df_data["precio_bolsa_sistema_daily"])

                else:
                    st.error(f"Error en la respuesta de FastAPI: {response.text}")

            except Exception as e:
                st.error(f"Error al generar predicciones: {e}")