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
import pandas as pd # Para poder leer tablas de datos
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import (TimeSeriesSplit, cross_validate)

# Creamos una constante con el nombre del csv
NOMBRE_CSV = "consumo_edificios_brisa.csv"
# Declaramos una semilla fija en forma de constante
RANDOM_STATE = 42
# Fijamos la fecha de corte, para predecir días futuros, hemos decidido que serán los ultimos 3 meses de 2025 (Octubre-Diciembre)
FECHA_CORTE_TEST = "2025-10-01"

"""
PRIMERO REALIZAMOS UNAS CONSULTAS BÁSICAS DE INFORMACIÓN
"""
# Lo primero es leer el archivo para que tenga los datos del csv
df = pd.read_csv(NOMBRE_CSV)

# Lo segundo es comprobar los tipos y el tamaño de los datos con df.shape y df.dtypes
print("\nFilas y columnas del archivo csv:", df.shape)

print("\nCuales son los tipos de los datos que vamos a tratar:", df.dtypes)

# Lo tercero es comprobar los periodos cubiertos usando .min() y .max()
print("\nPeriodo cubierto:", df["fecha"].min(), "hasta", df["fecha"].max())

# Vamos a consultar también el número de edificios que gestiona Brisa. Vamos a usar nunique que cuenta los valores distintos de una columna
print("\nNúmero de edificios diferentes:", df["id_edificio"].nunique())

# Por último vamos a consultar los valores ausentes por cada columna
print("\nValores ausente por columna:", df.isna().sum())

"""
AUDITORIA DE LAS VARIABLES EXCLUIDAS y CONSERVADAS
"""
# Declaramos una constante de cual es la variable objetivo
COLUMNA_OBJETIVO = "consumo_kwh"

# Creamos un diccionario que contrenda de clave las variables y de valor el motivo del porque hemos excluido dichas variables
MOTIVO_VARIABLES_EXCLUIDAS = {
    "id_registro": "No tiene relación con el objetivo y no es relevante.",
    "email_responsable": "Es un dato personal y por la ley de protección de datos y se excluye, aparte de no aportar nada para el análisis.",
    "referencia_catastral": "Por ley de protección de datos no podemos usar esta variable en el análisis, aparte de que ya disponemos de un identificador como id_edificio para identificar cada edificio.",
    "fecha": "Marca temporal usada para ordenar y partir los datos, no como rasgo directo en X.",
    "consumo_dia_anterior_kwh": "Este dato está excluido ya que el contador cierra a las 23:59 del día anterior, como Brisa predice a las 20:00 ese dato no existe para entonces.",
    "temp_real_media": "Lo hemos excluido ya que la estación meteorológica lo mide en el día que queremos anticipar por lo que no nos sirve de nada.",
    "consumo_semana_kwh": "Está excluida ya que este dato sólo se proporciona al cerrar la semana, por lo que nos incluye tanto el día que hemos predicho como los días posteriores y eso no nos sirve."
}

print("\nVariables excluidas y su motivo:")
for col, motivo in MOTIVO_VARIABLES_EXCLUIDAS.items():
    print(f"- {col}: {motivo}")