# Alcance y limitaciones

## Población y periodo

Los datos corresponden a clientes de tarjeta de crédito de Taiwán. La variable objetivo recoge el incumplimiento del mes siguiente según la definición del conjunto de UCI. Aplicar el modelo a otro país, producto o institución requeriría validación en esa población.

El entrenamiento y la prueba son particiones aleatorias del mismo conjunto. El PSI de 0,0011 describe la diferencia entre las distribuciones de predicciones de esas dos particiones. Para evaluar deterioro en el tiempo se necesitan observaciones de un periodo posterior. [Cálculo publicado del PSI](../outputs/tables/score_psi_summary.csv)

## Datos y decisiones del modelo

- La base contiene seis meses de historial y carece de información completa sobre ingresos, empleo y entorno macroeconómico. La interpretación de los coeficientes queda condicionada a las variables disponibles.
- Algunos códigos de educación y estado civil tienen significado incierto. Agruparlos permite conservar esas observaciones, aunque limita la interpretación de las categorías resultantes.
- La utilización del crédito se acota con el percentil 99 del entrenamiento. Cambiar ese criterio podría alterar la influencia de los casos extremos.
- La regresión logística sirve como referencia interpretable. El repositorio reproduce el modelo final. Una comparación con otros algoritmos o una búsqueda más amplia de hiperparámetros requerirían experimentos adicionales.

## Interpretación y uso

La calibración global puede ocultar diferencias dentro de los deciles o entre subgrupos. El modelo incluye variables demográficas y el análisis publicado no evalúa su desempeño por subgrupo. [Calibración por decil](../outputs/tables/final_test_decile_summary.csv)

El alcance es educativo. Para tomar decisiones de crédito, fijar límites o calcular provisiones harían falta validaciones adicionales de población, estabilidad, equidad y uso previsto.
