import os
import pandas as pd
from sqlalchemy import create_engine

# Prefect
from prefect import flow, task

@task
def cargar_variables_entorno():

    # Variables de Entorno de la base de datos
    db_user = "postgres"
    db_password = "postgres"
    db_host = "localhost"
    db_port = "5432"
    db_name_raw = "sistema_electrico_raw"
    db_name_transformed = "sistema_electrico_features"


    return {
        "local_raw": {
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port,
            "name": db_name_raw
        },
         "local_transformed": {
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port,
            "name": db_name_transformed
        }
    }


@task
def crear_conexion_bd(db_params: dict):
    """
    Crea y retorna un SQLAlchemy engine con los parámetros dados.
    """
    engine = create_engine(
        f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["name"]}',
        echo=False
    )
    return engine


@task
def sql_to_df(engine, sql_query: str) -> pd.DataFrame:
    """
    Ejecuta una consulta SQL sobre 'engine' y retorna un DataFrame.
    """
    df = pd.read_sql_query(sql_query, con=engine)
    return df


@task
def calcular_suma_diaria(df: pd.DataFrame, col_prefix="Values_Hour") -> pd.DataFrame:
    """
    Suma diaria de las columnas 'Values_Hour01' a 'Values_Hour24' y retorna un DF con una sola columna
    'X_daily'. El nombre se infiere del df si es para generacion, perdidas, etc.
    """
    # Suponiendo que la columna 'Date' está en el df y hay 'Values_HourXX'
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Encontrar las columnas que van de Values_Hour01 a Values_Hour24, independientemente
    # de mayúsculas, minúsculas, etc.
    hour_cols = [c for c in df.columns if c.lower().startswith(col_prefix.lower())]
    # Nombre base: si no hay otra referencia, la llamaremos 'col_daily'
    name_for_daily = "col_daily"

    # Heurística: si el DF proviene de una tabla "generacion_sistema",
    # podemos renombrar a "generacion_sistema_daily", etc.
    # o lo adaptamos con una variable externa. Por simplicidad,
    # haremos un rename manual más adelante en cada caso donde se requiera.

    df[name_for_daily] = df.loc[:, hour_cols].sum(axis=1)
    return df[[name_for_daily]]


@task
def calcular_promedio_diario(df: pd.DataFrame, col_prefix="Values_Hour") -> pd.DataFrame:
    """
    Calcula el promedio diario (mean) de las columnas 'Values_Hour01'...'Values_Hour24'.
    Se utiliza para Disponibilidad Real, por ejemplo.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    hour_cols = [c for c in df.columns if c.lower().startswith(col_prefix.lower())]
    df['col_daily'] = df.loc[:, hour_cols].mean(axis=1)
    return df[['col_daily']]


@task
def df_rename_col(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    """
    Renombra una columna en el DataFrame.
    """
    return df.rename(columns={old_name: new_name})


@task
def df_only_col(df: pd.DataFrame, col_name: str, new_name: str = None) -> pd.DataFrame:
    """
    Toma solo una columna de un DF y opcionalmente la renombra.
    """
    df = df[[col_name]]
    if new_name:
        df = df.rename(columns={col_name: new_name})
    return df


@task
def unir_por_date_outer(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Hace un merge outer entre df1 y df2 usando el índice "Date" o la columna "Date".
    Asume que ambos DF están indexados por 'Date' o la tienen como columna.
    """
    # Si tienen 'Date' como índice, podemos usar join
    if df1.index.name == 'Date' and df2.index.name == 'Date':
        return df1.join(df2, how='outer')
    else:
        # Sino, asume que 'Date' es columna
        return pd.merge(df1, df2, on="Date", how="outer")


@task
def imputar_nulos_cero(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Reemplaza NaN con 0 en las columnas indicadas.
    """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


@task
def exportar_csv(df: pd.DataFrame, path: str) -> None:
    """
    Exporta el DataFrame a CSV.
    """
    df.to_csv(path)
    print(f"CSV exportado en: {path}")


@task
def exportar_a_bd(df: pd.DataFrame, engine, table_name: str):
    """
    Exporta el DataFrame a la base de datos usando 'to_sql'.
    """
    df.to_sql(table_name, engine, if_exists='replace', index=True)
    print(f"Tabla '{table_name}' exportada correctamente.")


# ============================================================================
# FLOW PRINCIPAL
# ============================================================================
@flow(name="Feature Engineering Pipeline")
def flujo_transformaciones_analitica():
    """
    Flow que orquesta la lógica completa de transformaciones_analitica.ipynb con Prefect.
    """

    # 1) Cargar credenciales
    creds = cargar_variables_entorno()

    # 2) Conexión BD local
    engine_local = crear_conexion_bd(creds["local_raw"])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LÓGICA PASO A PASO (replicando la del notebook)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # --------------------------------------------------------------------------
    # (1) Generación Sistema
    # --------------------------------------------------------------------------
    q_generacion_sistema = """SELECT * FROM generacion_sistema"""
    df_gs = sql_to_df(engine_local, q_generacion_sistema)
    df_gs = calcular_suma_diaria(df_gs)
    df_gs = df_rename_col(df_gs, "col_daily", "generacion_sistema_daily")

    # --------------------------------------------------------------------------
    # (2) Generación Ideal
    # --------------------------------------------------------------------------
    q_generacion_ideal = """SELECT * FROM generacion_ideal_sistema"""
    df_gi = sql_to_df(engine_local, q_generacion_ideal)
    df_gi = calcular_suma_diaria(df_gi)
    df_gi = df_rename_col(df_gi, "col_daily", "generacion_ideal_sistema_daily")

    # Unión inicial
    df_consolidated = unir_por_date_outer(df_gs, df_gi)

    # --------------------------------------------------------------------------
    # (3) Pérdidas del Sistema
    # --------------------------------------------------------------------------
    q_perdidas = """SELECT * FROM perdidas_sistema"""
    df_ps = sql_to_df(engine_local, q_perdidas)
    df_ps = calcular_suma_diaria(df_ps)
    df_ps = df_rename_col(df_ps, "col_daily", "perdidas_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_ps)

    # --------------------------------------------------------------------------
    # (4) Volumen Útil
    # --------------------------------------------------------------------------
    q_vol_util = """SELECT * FROM volumen_util_energia_sistema"""
    df_vu = sql_to_df(engine_local, q_vol_util)
    df_vu['Date'] = pd.to_datetime(df_vu['Date'])
    df_vu = df_vu.set_index('Date')
    df_vu = df_only_col(df_vu, "Value", "volumen_util_energia_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_vu)

    # --------------------------------------------------------------------------
    # (5) Emisiones CO2
    #    Se realiza un group by y sum en la consulta original, la replicamos:
    # --------------------------------------------------------------------------
    q_emisiones = """
    SELECT
        sum("Values_Hour01") as Values_Hour01,
        sum("Values_Hour02") as Values_Hour02,
        sum("Values_Hour03") as Values_Hour03,
        sum("Values_Hour04") as Values_Hour04,
        sum("Values_Hour05") as Values_Hour05,
        sum("Values_Hour06") as Values_Hour06,
        sum("Values_Hour07") as Values_Hour07,
        sum("Values_Hour08") as Values_Hour08,
        sum("Values_Hour09") as Values_Hour09,
        sum("Values_Hour10") as Values_Hour10,
        sum("Values_Hour11") as Values_Hour11,
        sum("Values_Hour12") as Values_Hour12,
        sum("Values_Hour13") as Values_Hour13,
        sum("Values_Hour14") as Values_Hour14,
        sum("Values_Hour15") as Values_Hour15,
        sum("Values_Hour16") as Values_Hour16,
        sum("Values_Hour17") as Values_Hour17,
        sum("Values_Hour18") as Values_Hour18,
        sum("Values_Hour19") as Values_Hour19,
        sum("Values_Hour20") as Values_Hour20,
        sum("Values_Hour21") as Values_Hour21,
        sum("Values_Hour22") as Values_Hour22,
        sum("Values_Hour23") as Values_Hour23,
        sum("Values_Hour24") as Values_Hour24,
        "Date"
    FROM "emisiones_CO2eq"
    GROUP BY "Date"
    ORDER BY "Date"
    """
    df_emi = sql_to_df(engine_local, q_emisiones)
    df_emi = calcular_suma_diaria(df_emi)
    df_emi = df_rename_col(df_emi, "col_daily", "emisiones_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_emi)

    # --------------------------------------------------------------------------
    # (6) Aportes Energía Sistema
    # --------------------------------------------------------------------------
    q_aportes = """SELECT * FROM aportes_energia_sistema"""
    df_ap = sql_to_df(engine_local, q_aportes)
    df_ap['Date'] = pd.to_datetime(df_ap['Date'])
    df_ap = df_ap.set_index('Date')
    df_ap = df_only_col(df_ap, "Value", "aportes_energia_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_ap)

    # --------------------------------------------------------------------------
    # (7) Aportes Energía Sistema Media Histórica
    # --------------------------------------------------------------------------
    q_aportes_hist = """SELECT * FROM aportes_energia_sistema_media_historica"""
    df_aph = sql_to_df(engine_local, q_aportes_hist)
    df_aph['Date'] = pd.to_datetime(df_aph['Date'])
    df_aph = df_aph.set_index('Date')
    df_aph = df_only_col(df_aph, "Value", "aportes_energia_mediaHist_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_aph)

    # --------------------------------------------------------------------------
    # (8) Capacidad Útil Energía
    # --------------------------------------------------------------------------
    q_cap_util = """SELECT "Value","Date" FROM capacidad_util_energia"""
    df_cap = sql_to_df(engine_local, q_cap_util)
    df_cap['Date'] = pd.to_datetime(df_cap['Date'])
    df_cap = df_cap.set_index('Date')
    df_cap = df_rename_col(df_cap, "Value", "capacidad_util_energia_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_cap)

    # --------------------------------------------------------------------------
    # (9) Demanda Real
    # --------------------------------------------------------------------------
    q_demanda = """SELECT * FROM dema_real"""
    df_dem = sql_to_df(engine_local, q_demanda)
    df_dem = calcular_suma_diaria(df_dem)
    df_dem = df_rename_col(df_dem, "col_daily", "demanda_real_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_dem)

    # --------------------------------------------------------------------------
    # (10) Exportaciones
    # --------------------------------------------------------------------------
    q_exports = """SELECT * FROM exportaciones"""
    df_exp = sql_to_df(engine_local, q_exports)
    df_exp = calcular_suma_diaria(df_exp)
    df_exp = df_rename_col(df_exp, "col_daily", "exportaciones_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_exp)

    # --------------------------------------------------------------------------
    # (11) Importaciones
    # --------------------------------------------------------------------------
    q_imports = """SELECT * FROM importaciones"""
    df_imp = sql_to_df(engine_local, q_imports)
    df_imp = calcular_suma_diaria(df_imp)
    df_imp = df_rename_col(df_imp, "col_daily", "importaciones_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_imp)

    # --------------------------------------------------------------------------
    # (12) Precio Bolsa
    #    Se promedia la suma de Values_Hour / 24
    # --------------------------------------------------------------------------
    q_pb = """SELECT * FROM precio_bolsa_nacional"""
    df_pb = sql_to_df(engine_local, q_pb)
    df_pb['Date'] = pd.to_datetime(df_pb['Date'])
    df_pb = df_pb.set_index('Date')
    hour_cols_pb = [c for c in df_pb.columns if c.lower().startswith("values_hour")]
    df_pb['precio_bolsa_sistema_daily'] = df_pb[hour_cols_pb].sum(axis=1).apply(lambda x: x / 24)
    df_pb = df_pb[['precio_bolsa_sistema_daily']]
    df_consolidated = unir_por_date_outer(df_consolidated, df_pb)

    # --------------------------------------------------------------------------
    # (13) Precio Escasez
    # --------------------------------------------------------------------------
    q_escasez = """SELECT * FROM precio_escasez"""
    df_pe = sql_to_df(engine_local, q_escasez)
    df_pe['Date'] = pd.to_datetime(df_pe['Date'])
    df_pe = df_pe.set_index('Date')
    df_pe = df_only_col(df_pe, "Value", "precio_escasez_sistema_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_pe)

    # --------------------------------------------------------------------------
    # (14) Consumo Combustible MBTU
    # --------------------------------------------------------------------------
    q_comb = """
    SELECT
        sum("Values_Hour01") AS Values_Hour01,
        sum("Values_Hour02") AS Values_Hour02,
        sum("Values_Hour03") AS Values_Hour03,
        sum("Values_Hour04") AS Values_Hour04,
        sum("Values_Hour05") AS Values_Hour05,
        sum("Values_Hour06") AS Values_Hour06,
        sum("Values_Hour07") AS Values_Hour07,
        sum("Values_Hour08") AS Values_Hour08,
        sum("Values_Hour09") AS Values_Hour09,
        sum("Values_Hour10") AS Values_Hour10,
        sum("Values_Hour11") AS Values_Hour11,
        sum("Values_Hour12") AS Values_Hour12,
        sum("Values_Hour13") AS Values_Hour13,
        sum("Values_Hour14") AS Values_Hour14,
        sum("Values_Hour15") AS Values_Hour15,
        sum("Values_Hour16") AS Values_Hour16,
        sum("Values_Hour17") AS Values_Hour17,
        sum("Values_Hour18") AS Values_Hour18,
        sum("Values_Hour19") AS Values_Hour19,
        sum("Values_Hour20") AS Values_Hour20,
        sum("Values_Hour21") AS Values_Hour21,
        sum("Values_Hour22") AS Values_Hour22,
        sum("Values_Hour23") AS Values_Hour23,
        sum("Values_Hour24") AS Values_Hour24,
        "Date"
    FROM "consumo_combustible_MBTU"
    GROUP BY "Date"
    ORDER BY "Date" ASC
    """
    df_comb = sql_to_df(engine_local, q_comb)
    df_comb = calcular_suma_diaria(df_comb)
    df_comb = df_rename_col(df_comb, "col_daily", "comsumo_combustible_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_comb)

    # --------------------------------------------------------------------------
    # (15) Disponibilidad Real
    # --------------------------------------------------------------------------
    q_disp = """
    SELECT
        sum("Values_Hour01") as Values_Hour01,
        sum("Values_Hour02") as Values_Hour02,
        sum("Values_Hour03") as Values_Hour03,
        sum("Values_Hour04") as Values_Hour04,
        sum("Values_Hour05") as Values_Hour05,
        sum("Values_Hour06") as Values_Hour06,
        sum("Values_Hour07") as Values_Hour07,
        sum("Values_Hour08") as Values_Hour08,
        sum("Values_Hour09") as Values_Hour09,
        sum("Values_Hour10") as Values_Hour10,
        sum("Values_Hour11") as Values_Hour11,
        sum("Values_Hour12") as Values_Hour12,
        sum("Values_Hour13") as Values_Hour13,
        sum("Values_Hour14") as Values_Hour14,
        sum("Values_Hour15") as Values_Hour15,
        sum("Values_Hour16") as Values_Hour16,
        sum("Values_Hour17") as Values_Hour17,
        sum("Values_Hour18") as Values_Hour18,
        sum("Values_Hour19") as Values_Hour19,
        sum("Values_Hour20") as Values_Hour20,
        sum("Values_Hour21") as Values_Hour21,
        sum("Values_Hour22") as Values_Hour22,
        sum("Values_Hour23") as Values_Hour23,
        sum("Values_Hour24") as Values_Hour24,
        "Date"
    FROM "disponibilidad_real"
    GROUP BY "Date"
    ORDER BY "Date" ASC
    """
    df_disp = sql_to_df(engine_local, q_disp)
    df_disp = calcular_promedio_diario(df_disp)
    df_disp = df_rename_col(df_disp, "col_daily", "disponibilidad_real_daily")
    df_consolidated = unir_por_date_outer(df_consolidated, df_disp)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Imputación de valores nulos (emisiones, exportaciones, importaciones)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df_consolidated = imputar_nulos_cero(
        df_consolidated,
        ["emisiones_sistema_daily", "exportaciones_sistema_daily", "importaciones_sistema_daily"]
    )

    # (En el notebook se grafica 'emisiones_sistema_daily' antes y después, 
    #  si quieres ver la gráfica, descomenta la siguiente línea)
    # graficar(df_consolidated, 'emisiones_sistema_daily', 'Ton CO2')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Exportar CSV
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """    out_csv_path = "./data/analitica_sistema_electrico_colombia.csv"
    exportar_csv(df_consolidated, out_csv_path)"""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Exportar a BD en AWS (si hay credenciales definidas)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    aws_creds = creds["local_transformed"]
    if (aws_creds["user"] and aws_creds["password"] and aws_creds["host"]
        and aws_creds["port"] and aws_creds["name"]):
        engine_aws = crear_conexion_bd(aws_creds)
        exportar_a_bd(df_consolidated, engine_aws, "sistema_electrico_features")
    else:
        print("No se encontraron credenciales completas para AWS. Se omite la exportación a AWS.")

    print("\n\n¡Proceso con Prefect completado!\n")



if __name__ == "__main__":
    flujo_transformaciones_analitica()