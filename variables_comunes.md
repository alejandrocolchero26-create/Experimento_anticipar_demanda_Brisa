# Contrato de Variables Comunes — `s06_experimento.py`

Acuerdo de nombres de variables entre **Persona 1** (Auditoría y Preprocesamiento) y **Persona 2** (Validación y Evaluación) para trabajar en paralelo sin conflictos.

---

## 1. Constantes de Configuración
Parámetros globales definidos al inicio del script:

* `RUTA_DATOS`: Ruta al archivo CSV (por defecto `"consumo_edificios_brisa.csv"`).
* `RANDOM_STATE`: Semilla fija en `42`.
* `COLUMNA_OBJETIVO`: `"consumo_kwh"`.
* `FECHA_CORTE_TEST`: Fecha de separación entre desarrollo y prueba (ej. `"2025-10-01"`).

---

## 2. Listas de Columnas (Puente Auditoría → Preprocesador)
Definidas por **Persona 1** tras auditar el diccionario y consumidas por ambos:

* `COLS_EXCLUIDAS`: Columnas descartadas por fuga temporal o privacidad (`id_registro`, `email_responsable`, `referencia_catastral`, `temp_real_media`, `consumo_semana_kwh`, `consumo_dia_anterior_kwh`, `consumo_kwh`).
* `COLS_NUMERICAS`: Variables continuas (`superficie_m2`, `ocupacion_prevista_pct`, `temp_prevista_max`, `temp_prevista_min`, `consumo_hace_7_dias_kwh` + variables extraídas de `fecha`, si aplica).
* `COLS_NOMINALES`: Categóricas sin orden (`id_edificio`, `tipo_edificio`, `dia_semana`).
* `COLS_ORDINALES`: Categóricas con jerarquía (`certificacion_energetica`).
* `COLS_BINARIAS`: Variables binarias (`festivo`).
* `FEATURES`: Lista completa de predictoras válidas (`COLS_NUMERICAS + COLS_NOMINALES + COLS_ORDINALES + COLS_BINARIAS`).

---

## 3. Conjuntos de Datos Separados (Puente Reserva → Validación)
Resultantes del corte temporal estricto (ordenados previamente por `fecha`):

* `df_desarrollo`: Subconjunto histórico anterior a `FECHA_CORTE_TEST`.
* `df_test`: Subconjunto posterior o igual a `FECHA_CORTE_TEST` (**bloqueado hasta el cierre**).
* `X_desarrollo`, `y_desarrollo`: Predictoras (`df_desarrollo[FEATURES]`) y objetivo (`df_desarrollo[COLUMNA_OBJETIVO]`) para validación cruzada.
* `X_test`, `y_test`: Predictoras y objetivo reservados para la consulta única final.

---

## 4. Objetos de Modelado y Métricas (Zona Persona 2)
Objetos generados durante la validación y evaluación final:

* `preprocesador`: Objeto `ColumnTransformer` sin ajustar (*unfitted*) entregado por **Persona 1**.
* `cv`: Estrategia/generador de particiones temporales sobre `df_desarrollo`.
* `pipeline_ridge`: `Pipeline` con `preprocesador` + `Ridge(alpha=1.0)`.
* `pipeline_baseline`: `Pipeline` con `preprocesador` + `DummyRegressor(strategy="median")`.
* `resultados_ridge`, `resultados_baseline`: Diccionarios devueltos por `cross_validate`.
* `mae_ridge`, `mae_baseline`: Arrays con el MAE por pliegue en **kWh positivos**.
* `mae_test`: Escalar con el error final único sobre `X_test` e `y_test`.

---

> **Nota de flujo:** Mientras Persona 1 finaliza los detalles del `ColumnTransformer`, Persona 2 puede programar y probar toda la lógica de validación y métricas usando un `preprocesador` provisional sencillo y sustituirlo al integrar.
