# Detección y Clasificación de Embarcaciones

Este repositorio contiene los códigos para la detección y clasificación automática de embarcaciones, basados en características acústicas extraídas de grabaciones de Monitoreo Acústico Pasivo (PAM). Los algoritmos de machine learning utilizados incluyen SVM, KNN, RF y DT.

Esta implementación forma parte de mi tesis de grado "Detección y Clasificación Automática de Embarcaciones por Parámetros Acústicos", donde se presenta un pipeline funcional y una comparación del rendimiento de estos algoritmos. El objetivo es contribuir al desarrollo de sistemas inteligentes para la detección de señales acústicas en extensas grabaciones producto de PAM en entornos marinos.

RESUMEN DE TESIS:

En el presente trabajo de tesis se desarrollan algoritmos de detección y clasificación de embarcaciones basados en características acústicas. Se busca proporcionar herramientas para automatizar el análisis en el monitoreo acústico pasivo (PAM) y sentar las bases para futuras mejoras en los sistemas propuestos.
La metodología empleada consta de tres etapas principales. En primer lugar, se realiza un estudio exhaustivo del comportamiento acústico de las embarcaciones y se crea una base de datos con diversas muestras. En segundo lugar, se obtienen y evalúan diferentes descriptores acústicos, tanto estandarizados como propuestos. Se destaca el parámetro "índice de tonalidad" por su capacidad para distinguir entre embarcaciones y no embarcaciones.
En la tercera etapa, se desarrollan y evalúan cuatro modelos de aprendizaje automático para la detección y clasificación de embarcaciones: Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest (RF) y Decision Tree (DT). Se calculan métricas de desempeño para los algoritmos de detección, obteniendo resultados satisfactorios en todos los modelos, con valores de F1 superiores a 0,95. Además, se evalúa la capacidad de generalización de los modelos en archivos de larga duración independientes de los conjuntos de datos utilizados para el entrenamiento, demostrando una buena capacidad para detectar embarcaciones en condiciones realistas.
En cuanto a la clasificación, se encontró que el modelo SVM obtiene los mejores resultados en todas las categorías, con valores de F1 superiores a 0,8, seguido por el modelo KNN con un rendimiento aceptable, valores de F1 superiores a 0,75.
