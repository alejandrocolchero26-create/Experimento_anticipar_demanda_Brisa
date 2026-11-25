import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# ==============================================================================
# BLOQUE PROVISIONAL (Sustituir por el código de Persona 1 al integrar)
# ==============================================================================
RUTA_DATOS = "consumo_edificios_brisa.csv"
COLUMNA_OBJETIVO = "consumo_kwh"
FECHA_CORTE_TEST = "2025-10-01"

# 1. Carga y ordenación temporal mínima
df_temp = pd.read_csv(RUTA_DATOS)
df_temp["fecha"] = pd.to_datetime(df_temp["fecha"])
df_temp = df_temp.sort_values(["fecha", "id_edificio"]).reset_index(drop=True)

# 2. Corte temporal para simular el contrato (desarrollo vs test)
df_desarrollo = df_temp[df_temp["fecha"] < FECHA_CORTE_TEST].copy()
df_test = df_temp[df_temp["fecha"] >= FECHA_CORTE_TEST].copy()

# 3. Selección rápida de 3 columnas numéricas legales (sin fuga) para probar
FEATURES_PROV = ["ocupacion_prevista_pct", "temp_prevista_max", "temp_prevista_min"]

X_desarrollo = df_desarrollo[FEATURES_PROV]
y_desarrollo = df_desarrollo[COLUMNA_OBJETIVO]

X_test = df_test[FEATURES_PROV]
y_test = df_test[COLUMNA_OBJETIVO]

# 4. Preprocesador de juguete (sin ajustar / unfitted)
# Incluye imputador porque temp_prevista_max/min pueden tener nulos según el diccionario
preprocesador = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

# ==============================================================================
# FIN DEL BLOQUE PROVISIONAL
# ==============================================================================


