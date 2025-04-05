import os
import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Configuración de Streamlit en modo oscuro
st.set_page_config(page_title="Dasboard Monitoreo", layout="wide")

# Credenciales de base de datos
db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "postgres")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME_TRANSFORMED", "sistema_electrico_features")

# Consulta SQL
query = """
SELECT
    s."Date" AS time, 
    s.precio_bolsa_sistema_daily AS real_value,
    p.predicted_precio_bolsa_sistema_daily AS predicted_value
FROM sistema_electrico_test_set s
JOIN predictions p 
    ON s."Date" = p.prediction_date
ORDER BY s."Date"
"""

# Conexión y carga de datos
@st.cache_data
def cargar_datos():
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

st.title("Dashboard Monitoreo Metricas ML")

def plot():
    df = cargar_datos()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Plot
    fig = px.line(
    df,
    y=['real_value', 'predicted_value'],
    title='Precio Bolsa del Sistema vs Predicción',
    color_discrete_sequence=['deepskyblue', 'orange']  # azul para la primera, naranja para la segunda
    )

    st.plotly_chart(fig, use_container_width=True)

    # Métricas
    rmse = np.sqrt(mean_squared_error(df["real_value"], df["predicted_value"]))
    r2 = r2_score(df["real_value"], df["predicted_value"])

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("R²", f"{r2:.3f}")


try:
    if st.button("reCargar"):
        plot()
    else:
        plot()
        
    

except Exception as e:
    st.error(f"Error al cargar datos: {e}")