"""
FORMULAR LA TAREA ANTES DE PROGRAMMAR

1 - Qué representa una fila y qué variable actúa como objetivo:
    - Cada fila representa el registro diario de un edificio en una fecha determinada con sus datos que conducen al consumo
    - La variable que actúa como objetivo es 'consumo_kwh' porque recoge la información sobre el consumo en kwh de un edificio durante todo el día
    
2 - En qué momento exacto se realiza la predicción y para qué periodo:
    - La realización de las predicciones se hace cada día a las 20:00 del día anterior al día que se quiere predecir
    - El periodo es para el día siguiente en todas sus horas (24 horas)
    
3 - Qué decisión de Brisa depende de esa predicción: 
    - La decision de planificar la climatización y la compra de energía de cada edificio para el día siguiente sin la necesidad de hacerlo manualmente
    
4 - Cómo se utilizaría el sistema en el futuro: sobre qué días y sobre qué edificios.
    - Se usará todos los días posteriores al historico que usamos para entrenar durante 24 horas.
    - Se usarán para todos los edificios que gestiona Brisa que estén en el historico que hemos usado para entrenar
"""

# Lo primero que haremos sera importar las librerias necesarias
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import (TimeSeriesSplit, cross_validate)
from sklearn.pipeline import Pipeline

# Creamos una constante con el nombre del csv
RUTA_DATOS = "consumo_edificios_brisa.csv"

# Declaramos una semilla fija en forma de constante
RANDOM_STATE = 42

# Declaramos una constante de cual es la variable objetivo
COLUMNA_OBJETIVO = "consumo_kwh"

# Fijamos la fecha de corte, para predecir días futuros, hemos decidido que serán los ultimos 3 meses de 2025 (Octubre-Diciembre)
FECHA_CORTE_TEST = "2025-10-01"

"""
Carga y Fase 1 de EDA
"""
# Lo primero es leer el archivo para que tenga los datos del csv
df = pd.read_csv(RUTA_DATOS)

# Lo segundo es comprobar los tipos y el tamaño de los datos con df.shape y df.dtypes
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print("\nCuales son los tipos de los datos que vamos a tratar:", df.dtypes)

# Lo tercero es comprobar los periodos cubiertos usando .min() y .max()
print("\nPeriodo cubierto:", df["fecha"].min(), "hasta", df["fecha"].max())

# Vamos a consultar también el número de edificios que gestiona Brisa. Vamos a usar nunique que cuenta los valores distintos de una columna
print("\nNúmero de edificios diferentes:", df["id_edificio"].nunique())

# Verificamos los registros duplicados
duplicados = df.duplicated(subset = ["id_edificio", "fecha"]).sum()
print(f"Registro de duplicados (id_edificio + fecha): {duplicados}")

# Por último vamos a consultar los valores ausentes por cada columna
print("\nValores ausente por columna:", df.isna().sum())

"""
Auditoría de las variables
"""
# Creamos un diccionario que contrenda de clave las variables y de valor el motivo del porque hemos excluido dichas variables
MOTIVOS_EXCLUSION = {
    "id_registro": "No tiene relación con el objetivo y no es relevante.",
    "email_responsable": "Es un dato personal y por la ley de protección de datos y se excluye, aparte de no aportar nada para el análisis.",
    "referencia_catastral": "Por ley de protección de datos no podemos usar esta variable en el análisis, aparte de que ya disponemos de un identificador como id_edificio para identificar cada edificio.",
    "fecha": "Marca temporal usada para ordenar y partir los datos, no como rasgo directo en X.",
    "consumo_dia_anterior_kwh": "Este dato está excluido ya que el contador cierra a las 23:59 del día anterior, como Brisa predice a las 20:00 ese dato no existe para entonces.",
    "temp_real_media": "Lo hemos excluido ya que la estación meteorológica lo mide en el día que queremos anticipar por lo que no nos sirve de nada.",
    "consumo_semana_kwh": "Está excluida ya que este dato sólo se proporciona al cerrar la semana, por lo que nos incluye tanto el día que hemos predicho como los días posteriores y eso no nos sirve."
}

# Lista con las variables excluidas
COLS_EXCLUIDAS = [
    "id_registro",
    "email_responsable",
    "referencia_catastral",
    "temp_real_media",
    "consumo_semana_kwh",
    "consumo_dia_anterior_kwh",
    "consumo_kwh"
]

print("\nVariables excluidas y su motivo:")
for col, motivo in MOTIVOS_EXCLUSION.items():
    print(f"- {col}: {motivo}")
    
# Variables predictoras que son válidas a as 20:00 del día anterior, clasificadas por tipos
COLS_NUMERICAS = ["superficie_m2", "ocupacion_prevista_pct", "temp_prevista_max", "temp_prevista_min", "consumo_hace_7_dias_kwh"]

COLS_NOMINALES = ["id_edificio", "tipo_edificio", "dia_semana"]

COLS_ORDINALES = ["certificacion_energetica"]

COLS_BINARIAS = ["festivo"]

ORDEN_CERTIFICACION = ["A", "B", "C", "D", "E", "F", "G"]

FEATURES = COLS_NUMERICAS + COLS_NOMINALES + COLS_ORDINALES + COLS_BINARIAS

"""
Reservar el conjunto de prueba
"""
# Primero, antes de dividir, vamos a ordenar de forma cronológica para garantizar la secuencia temporal
df_ordenado = df.sort_values(["fecha", "id_edificio"]).reset_index(drop = True)

mascara_test = df_ordenado["fecha"] >= FECHA_CORTE_TEST
df_desarrollo = df_ordenado.loc[~mascara_test].copy().reset_index(drop = True)
df_test = df_ordenado.loc[mascara_test].copy().reset_index(drop=True)

X_desarrollo = df_desarrollo[FEATURES].copy()
y_desarrollo = df_desarrollo[COLUMNA_OBJETIVO].copy()

# X_test e y_test quedan cerrados
X_test = df_test[FEATURES].copy()
y_test = df_test[COLUMNA_OBJETIVO].copy()

print("Reserva de test")
print(f"Desarrollo (df_desarrollo): {len(df_desarrollo)} filas ({len(df_desarrollo)/len(df_ordenado):.1%}) | "
      f"Periodo: {df_desarrollo['fecha'].min()} hasta {df_desarrollo['fecha'].max()}")

print(f"Prueba (df_test): {len(df_test)} filas ({len(df_test)/len(df_ordenado):.1%}) | "
      f"Periodo: {df_test['fecha'].min()} hasta {df_test['fecha'].max()}")

"""
Fase 2 del EDA
"""
# Vamos a ir tratando cada una de las dimensiones del EDA
# Paso de dimension 1 - Completitud en desarrollo
print("\n1 - Completitud: Valores ausentes en desarrollo:")
print(df_desarrollo[FEATURES + [COLUMNA_OBJETIVO]].isna().sum().to_string())

# Paso de dimension 2 - Validez: Categorias y Binarias
print("\n2- Validez: Categorías observadas en desarrollo:")
for col in COLS_NOMINALES+ COLS_ORDINALES + COLS_BINARIAS:
    print(f"- {col}: {sorted(df_desarrollo[col].dropna().unique())}")

# Paso de dimension 3 - Consistencia física y lógica
# Comprobamos en cuanto a consistencia si existe una temperatura prevista minima que sea mayor que la maxima
incoherencias_temp = (df_desarrollo["temperatura_prevista_min"] > df_desarrollo["temperatura_prevista_max"]).sum()
# Comprobamos si hay superficies por metro cuadrado que sea menor o igual a 0
superficie_invalida = (df_desarrollo["superficie_m2"] <= 0).sum()
# Comprobamos si hay una ocupacion prevista entre 0 y 100
ocupacion_fuera_rango = ((df_desarrollo["ocupacion_prevista_pct"] < 0) | (df_desarrollo["ocupacion_prevista_pct"] > 100)).sum()

print("\n3 - Consistencia: Comprobaciones de rangos lógicos:")
print(f"-  Días con temp_prevista_min > temp_prevista_max: {incoherencias_temp}")
print(f"- Registros con superficie_m2 <= 0: {superficie_invalida}")
print(f"- Registros con ocupacion_prevista_pct fuera de [0, 100]: {ocupacion_fuera_rango}")

# Paso de dimension 4 - Extremos y escalas numéricas
print("\n4 - Extremos y dispersion: Resumen numérico en desarrollo:")
print(df_desarrollo[COLS_NUMERICAS + [COLUMNA_OBJETIVO]].describe().round(2).to_string())

# Paso de dimension 5 - Correlación lineal con la variable objetivo
print("\n5 - Correlaciones: Correlacion con consumo_kwh en desarrollo:")
print(df_desarrollo[COLS_NUMERICAS + [COLUMNA_OBJETIVO]].corr()[COLUMNA_OBJETIVO].round(3).to_string())

"""
Diseñar la validación dentro de desarrollo

- Estrategia de entrenamiento: hemos optado por usar TimeSeriesSplit(n_splits = 5) con ventana creciente

- Que situacion imita: cada pliegue entrenará con el historico pasado y valida sobre un bloque posterior a ese tiempo, simulando como Brisa entrenará con los datos acumulados para anticipar dias futuros

- Que riesgo evita: evita la fuga temporal, de usar días futuros para predecir el pasado y la sobreestimacion por autocorrelacion.

- Requisitos previos: requiere que df_desarrollo esté ordenado de forma cronológica por fecha.

- Limite declarado: al haber múltiples edificios por día y cortar TimeSeriesSplit por número fijo de filas, la frontera exacta de un pliegue puede
dividir una misma fecha entre entrenamiento y validación, y los primeros pliegues cubren menos ciclos estacionales completos que los últimos.
"""

cv = TimeSeriesSplit(n_splits=5)

print("\n" + "=" * 75)
print("VALIDACIÓN TEMPORAL EN DESARROLLO (TimeSeriesSplit)")
print("=" * 75)
for pliegue, (idx_train, idx_val) in enumerate(cv.split(X_desarrollo), start=1):
    f_train_ini = df_desarrollo.loc[idx_train, "fecha"].min()
    f_train_fin = df_desarrollo.loc[idx_train, "fecha"].max()
    f_val_ini = df_desarrollo.loc[idx_val, "fecha"].min()
    f_val_fin = df_desarrollo.loc[idx_val, "fecha"].max()
    print(f"Pliegue {pliegue}: Train={len(idx_train)} filas ({f_train_ini}..{f_train_fin}) -> "
          f"Val={len(idx_val)} filas ({f_val_ini}..{f_val_fin})")
"""


# Contrucción del pipeline del modelo y baseline:

# Como completar los valores ausentes de las variables numéricas y categóricas

# "Rellenar"


# variables son nominales, cuáles tienen un orden real y cómo se codifican

# "Rellenar"

# Qué ocurre si aparece una categoría que no estaba presente al ajustar el preprocesador

# "Rellenar"


# qué variables necesitan escalado.

# Rellenar


# Encadenamos el preprocesador sin ajustar con los estimadores fijados por la empresa.
# No modificamos hiperparámetros de Ridge para mantener el experimento controlado.

pipeline_ridge = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("modelo", Ridge(alpha=1.0))
])

# Baseline equivalente: ignora las variables predictoras y predice siempre la mediana
# del consumo de entrenamiento. Sirve para comprobar si Ridge aporta valor real.
pipeline_baseline = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("modelo", DummyRegressor(strategy="median"))
])