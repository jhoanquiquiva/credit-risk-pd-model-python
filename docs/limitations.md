# Limitaciones del modelo PD

## 1. Naturaleza del dataset

El proyecto utiliza el conjunto público Default of Credit Card Clients de UCI.

Los datos corresponden a clientes de tarjetas de crédito de Taiwán y no representan necesariamente el comportamiento crediticio de una entidad financiera colombiana.

Por esta razón, los resultados no deben generalizarse directamente a otra institución, producto o país.

## 2. Validación no temporal

La división entre entrenamiento, validación y prueba fue aleatoria y estratificada.

Aunque la muestra de prueba fue independiente del entrenamiento, no constituye una validación out-of-time. Por tanto, no permite demostrar que el modelo mantendría su desempeño en periodos futuros.

## 3. Alcance de la variable objetivo

La variable objetivo identifica incumplimiento durante el mes siguiente según la definición incluida en el dataset.

No se desarrolló una definición regulatoria propia de default ni una ventana de desempeño adaptada a una entidad financiera específica.

## 4. Interpretación no causal

Las relaciones identificadas representan asociaciones observadas.

Los coeficientes de la regresión logística no demuestran que una variable cause el incumplimiento. También pueden existir variables omitidas, relaciones indirectas o factores económicos no incluidos.

## 5. Variables disponibles

El modelo utiliza únicamente la información contenida en el dataset.

No incorpora, entre otros:

- variables macroeconómicas;
- información de centrales de riesgo externas;
- ingresos o capacidad de pago completa;
- estabilidad laboral;
- información transaccional adicional;
- comportamiento posterior a la fecha de observación.

## 6. Códigos categóricos no documentados

Algunos códigos de educación y estado civil no estaban documentados claramente.

Estos valores fueron agrupados en categorías Other/Unknown para evitar segmentos pequeños y mantener todas las observaciones. Esta decisión puede afectar la interpretación de dichos coeficientes.

## 7. Tratamiento de valores extremos

La utilización promedio del crédito fue limitada al percentil 99 para reducir la influencia de valores extremos.

Se conservaron la variable original y banderas específicas para mantener trazabilidad. No obstante, otro criterio de tratamiento podría producir resultados diferentes.

## 8. Selección del modelo

El modelo final fue seleccionado comparando desempeño y parsimonia en la muestra de validación.

No se realizó una búsqueda exhaustiva de hiperparámetros ni una comparación completa con modelos como árboles, random forest o gradient boosting.

La regresión logística fue priorizada por su interpretabilidad y utilidad como línea base.

## 9. PSI

El PSI fue calculado entre entrenamiento y prueba y obtuvo un valor cercano a 0,0012.

Como ambas muestras proceden de una división aleatoria del mismo dataset, este resultado demuestra el procedimiento de cálculo, pero no constituye evidencia de estabilidad temporal real.

## 10. Uso previsto

El proyecto tiene finalidad educativa y de portafolio.

No debe utilizarse para:

- aprobar o rechazar solicitudes reales;
- asignar límites de crédito;
- calcular capital regulatorio;
- estimar provisiones contables;
- tomar decisiones financieras sin validación adicional.

Antes de una implementación real sería necesario realizar validación independiente, revisión regulatoria, análisis de sesgos, pruebas de estabilidad temporal y monitoreo periódico.
