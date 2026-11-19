# Metodología del modelo PD

## 1. Objetivo

El proyecto desarrolla un modelo exploratorio de probabilidad de incumplimiento (PD) utilizando datos públicos del conjunto Default of Credit Card Clients de UCI.

La variable objetivo indica si el cliente incumplió el pago durante el mes siguiente:

- `default = 1`: cliente en incumplimiento.
- `default = 0`: cliente sin incumplimiento.

La unidad de análisis es el cliente.

## 2. Calidad de datos

La base contiene 30.000 clientes y 25 variables originales.

Se realizaron controles sobre:

- filas duplicadas;
- unicidad del identificador;
- valores faltantes;
- integridad de la variable objetivo;
- rangos numéricos;
- códigos categóricos no documentados.

No se encontraron clientes duplicados ni valores faltantes.

Los códigos no documentados de educación y estado civil fueron agrupados dentro de categorías Other/Unknown, conservando también las variables originales para mantener trazabilidad.

## 3. Construcción de variables

Se construyeron variables que resumen seis meses de comportamiento financiero.

### Comportamiento de pago

- presencia de atraso reciente;
- presencia de atraso severo reciente;
- cantidad de meses con atraso;
- cantidad de meses con atraso severo;
- máximo atraso observado;
- cantidad de meses con saldo facturado positivo y pago igual a cero.

### Actividad financiera

- facturación promedio;
- pago promedio;
- utilización promedio del límite de crédito.

Los valores negativos y extremos de utilización fueron identificados mediante banderas. La variable continua fue limitada al percentil 99 para reducir la influencia de observaciones extremas, sin eliminar clientes ni modificar el valor original.

## 4. División de muestras

La población se dividió de manera aleatoria y estratificada:

- entrenamiento: 60%, 18.000 clientes;
- validación: 20%, 6.000 clientes;
- prueba: 20%, 6.000 clientes.

La estratificación permitió conservar una tasa de incumplimiento cercana al 22,12% en las tres muestras.

La base no contiene una fecha adecuada para construir una validación out-of-time real. Por tanto, la muestra de prueba representa una evaluación independiente aleatoria, no temporal.

## 5. Preprocesamiento

Las variables se trataron según su naturaleza:

- variables categóricas: one-hot encoding;
- variables numéricas: estandarización;
- variables binarias: conservación como 0 y 1.

El preprocesamiento fue ajustado exclusivamente con la muestra de entrenamiento para evitar fuga de información.

## 6. Modelo base

Se utilizó regresión logística por su:

- interpretabilidad;
- capacidad de producir probabilidades individuales;
- facilidad de validación;
- utilidad como modelo de referencia en riesgo de crédito.

No se realizó balanceo artificial de clases, ya que el objetivo era conservar la prevalencia observada de incumplimiento.

## 7. Selección del modelo final

Se compararon:

- un modelo completo de 17 variables;
- un modelo parsimonioso de 12 variables.

El modelo parsimonioso presentó una reducción pequeña de desempeño en validación, pero ofreció menor redundancia, mayor facilidad de explicación y una brecha más pequeña entre entrenamiento y validación.

Por esta razón fue seleccionado como modelo final.

## 8. Métricas de validación

El modelo fue evaluado mediante:

- AUC;
- Gini;
- KS;
- tasa observada frente a PD promedio;
- razón observado/esperado (O/E);
- Brier Score;
- Brier Skill Score;
- calibración por deciles;
- PSI del score.

## 9. Evaluación final

En la muestra de prueba independiente, el modelo obtuvo aproximadamente:

- AUC: 0,7431;
- Gini: 0,4862;
- KS: 0,3852;
- default observado: 22,1167%;
- PD promedio: 22,1533%;
- O/E: 0,9983;
- Brier Score: 0,1396;
- Brier Skill Score: 18,95%.

La tasa de incumplimiento aumentó desde 6,50% en el decil de menor riesgo hasta 69,17% en el decil de mayor riesgo.

## 10. Estabilidad

El PSI entre entrenamiento y prueba fue aproximadamente 0,0012, indicando distribuciones de score muy similares.

Como ambas muestras proceden de una división aleatoria del mismo dataset, este resultado demuestra la metodología de monitoreo, pero no constituye evidencia de estabilidad temporal.

## 11. Interpretación

Los resultados representan asociaciones observadas en los datos.

Los coeficientes del modelo no deben interpretarse como relaciones causales. El proyecto tiene finalidad educativa y de portafolio y no constituye un modelo regulatorio listo para producción.
