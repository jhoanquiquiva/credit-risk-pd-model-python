# Metodología del modelo de incumplimiento

## Población y variable objetivo

La base de UCI contiene una fila por cliente de tarjeta de crédito. De 30.000 clientes, 6.636 incumplieron en el mes siguiente (`default = 1`). La auditoría revisa duplicados, identificadores, faltantes y categorías inesperadas. El archivo publicado registra los conteos de cada control. [Controles de calidad](../outputs/tables/data_quality_checks.csv) · [Distribución del objetivo](../outputs/tables/target_distribution.csv)

## Preparación de las variables

Los seis meses de historial se resumen a nivel cliente. Entre las variables finales están el número de meses con atraso severo, el pago medio y los meses con factura positiva y pago igual a cero. Los códigos de educación y estado civil sin categoría documentada se agrupan antes del entrenamiento. [Preparación de datos](../src/03_prepare_modeling_dataset.py)

La división aleatoria usa la semilla 42 y conserva la proporción de incumplimientos. El percentil 99 de utilización se calcula **después** de dividir la muestra, usando los índices de entrenamiento. Ese mismo límite se aplica a todos los clientes. Para la variable continua se acotan los valores entre cero y el límite superior. Dos indicadores conservan la información de utilización negativa o extrema. [Resumen de las muestras](../outputs/tables/sample_split_summary.csv) · [Controles de la base preparada](../outputs/tables/modeling_dataset_validation.csv)

## Ajuste y selección

Durante el desarrollo se consideraron una regresión de 17 variables y una especificación reducida. El código publicado reproduce el ajuste final de **12 variables**. En validación, este modelo obtuvo un AUC de 0,7563. Las métricas y los coeficientes que se pueden comprobar aquí corresponden a la versión de 12 variables. [Entrenamiento](../src/04_train_final_pd_model.py) · [Métricas de validación](../outputs/tables/final_model_validation_metrics.csv)

El preprocesamiento codifica las variables categóricas, estandariza las numéricas y conserva los indicadores binarios. Se ajusta junto con la regresión usando los 18.000 clientes de entrenamiento. Las probabilidades de validación y prueba se obtienen con ese ajuste. Los coeficientes describen asociaciones condicionadas a las demás variables. [Coeficientes y razones de momios](../outputs/tables/final_model_coefficients.csv)

## Evaluación en prueba

Sobre los 6.000 clientes de prueba, el AUC fue **0,7431**, el Gini **0,4861** y el KS **0,3852**. La probabilidad media estimada fue 22,1526 % y la tasa observada 22,1167 %, lo que da una razón observado/esperado de 0,9984. [Métricas de prueba](../outputs/tables/final_test_metrics.csv)

La puntuación de Brier mide el error cuadrático medio de las probabilidades. Fue 0,1396 para el modelo y 0,1723 al predecir para todos la tasa de incumplimiento del entrenamiento. La reducción relativa del error fue del 18,95 %. [Cálculo de las métricas y de la referencia](../src/05_evaluate_final_model.py)

Al ordenar a los clientes por probabilidad estimada, el incumplimiento observado fue del 6,50 % en el primer decil y del 69,17 % en el último. El detalle por decil permite detectar diferencias de calibración que se compensan en el promedio general. [Resultados por decil](../outputs/tables/final_test_decile_summary.csv)

## Distribución de las predicciones

Para calcular el índice de estabilidad poblacional (PSI), se definen las bandas con las predicciones de entrenamiento y se comparan las proporciones de entrenamiento y prueba en cada banda. El PSI fue **0,0011**. Ambas muestras proceden del mismo periodo, de modo que la estabilidad en el tiempo sigue sin evaluarse. [Detalle por banda](../outputs/tables/score_psi_by_band.csv) · [PSI total](../outputs/tables/score_psi_summary.csv)
