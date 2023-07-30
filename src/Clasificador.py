import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

def main():

    print("Iniciando programa...")
    # datos_prueba son segmentos de 15s de archivos externos a los datasets
    # usados para entrenar para evaluar la generalizacion de los modelos
    X,y,param = dataSetGen()

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Detectores basados en sklearn
    clasificadores(X_train, X_test, y_train, y_test,param)

    print("Programa finalizado.")

def dataSetGen():
    print("Procesando datos...")

    # Leer los datos de los archivos excel
    datos_grupoGrande = pd.read_excel("D:\Tesis\Archivos Tesis FINAL\Parámetros\Grandes.xlsx")
    datos_grupoMediana = pd.read_excel("D:\Tesis\Archivos Tesis FINAL\Parámetros\Medianas.xlsx")
    datos_grupoPequeña = pd.read_excel("D:\Tesis\Archivos Tesis FINAL\Parámetros\Pequeñas.xlsx")
    datos_grupoNE = pd.read_excel("D:\Tesis\Archivos Tesis FINAL\Parámetros\RF.xlsx")

    datos_grupoGrande,param = aconDatos(datos_grupoGrande)
    datos_grupoMediana,param = aconDatos(datos_grupoMediana)
    datos_grupoPequeña,param = aconDatos(datos_grupoPequeña)
    datos_grupoNE,param = aconDatos(datos_grupoNE)

    # Genera y mezcla muestras de los grupos
    datos_grupoGrande = shuffle(datos_grupoGrande)
    datos_grupoMediana = shuffle(datos_grupoMediana)
    datos_grupoPequeña = shuffle(datos_grupoPequeña)
    datos_grupoNE = shuffle(datos_grupoNE)
    # Selecciona las filas correspondientes a 80 valores con índices aleatorios (para balancear categorías)
    datos_grupoNE = datos_grupoNE[np.random.choice(240, size=80, replace=False), :]

    # Crear la variable "X" de características y la variable "y" de etiquetas
    X = np.concatenate([datos_grupoGrande,datos_grupoMediana,datos_grupoPequeña,datos_grupoNE], axis=0)

    y = np.concatenate((np.repeat(3, datos_grupoGrande.shape[0]),
                        np.repeat(2, datos_grupoMediana.shape[0]),
                        np.repeat(1, datos_grupoPequeña.shape[0]),
                        np.repeat(0, datos_grupoNE.shape[0])), axis=0)

    # GE = 3 ; ME = 2; PE = 1; NE = 0 ;

    # Normalizar los datos
    sc = StandardScaler()
    X[:,:7] = sc.fit_transform(X[:,:7]) # normaliza columnas de descriptores individuales (agrupados hasta la columna 7)

    return X, y, param

def aconDatos(X):

    #conserva y ordena las etiquetas de los headers para que sean consistentes el dataset despues de procesarlo
    etiquetas = X.columns.tolist()[1:]
    et2 = etiquetas[:5]
    et2.append(etiquetas[8])
    et2.append(etiquetas[9])
    et2.append(etiquetas[5])
    et2.append(etiquetas[6])
    et2.append(etiquetas[7])
    et2.append(etiquetas[10])
    et2.append(etiquetas[11])
    etiquetas = et2

    # Elimina nombres de archivos del df
    datos = np.delete(X.to_numpy(), 0, axis=1)

    datos_2 = datos.copy()
    j = 0
    cantCol = [] #almacenará la cantidad de columnas de cada parámetro
    cantCol2 = []

    # Separa columnas con arrays en columnas individuales para su procesamiento
    for i in range(datos_2.shape[1]):
        aux = datos_2[:,i]

        if isinstance(aux[0], str):
            # Convierto la columna donde esté el array con formato string en una matriz numérica para reemplazar la columna

            # convertir cada fila de strings en una lista numérica
            matriz_numerica = []
            for fila in aux:
                fila = fila.strip("[]")  # eliminar corchetes

                fila = fila.replace("\n", "")  # eliminar saltos de línea
                lista_numerica = [float(valor) for valor in fila.replace(',', '').split()] # convertir a float y crear lista
                if len(lista_numerica) > 1: # normaliza descriptores vectoriales
                    A = np.asarray(lista_numerica)
                    B = A + np.abs(np.min(A))
                    lista_numerica = (B / np.max(B)).tolist()
                matriz_numerica.append(lista_numerica)

            # convertir la matriz en un array NumPy
            matriz_numerica = np.array(matriz_numerica)

            # Ahora se eliminará la columna con arrays en formato string y se agregarán las columnas de la matriz_numerica

            # Eliminamos la columna existente de la matriz original
            datos = np.delete(datos, i-j, axis=1)
            j += 1 # j sirve para ajustar el índice de la columna al eliminar una columna de datos

            # Unimos el dataframe original con el nuevo dataframe usando pd.concat() en el eje de las columnas
            datos = np.concatenate((datos, matriz_numerica), axis=1)

            cantCol2.append(len(lista_numerica))

        else:
            cantCol.append(1)

    cantCol.extend(cantCol2)

    # GESTIÓN DE OUTLIERS
    columnas = datos.shape[1]

    # Itera sobre cada columna
    for i in range(columnas):
        columna = datos[:, i]
        Q1 = np.percentile(columna, 25)
        Q2 = np.percentile(columna, 50)
        Q3 = np.percentile(columna, 75)
        IQR = Q3 - Q1

        # Calcula los límites para determinar los outliers
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Identifica los outliers en la columna
        outliers = columna[(columna < limite_inferior) | (columna > limite_superior)]
        if len(outliers)<round(X.shape[0]*0.05): #si los outliers son menos que un 5% del dataset, reemplaza por mediana
            columna[columna < limite_inferior] = Q2
            columna[columna > limite_superior] = Q2

        # si son muchos los outliers,los deja ya que no se pueden considerar outliers sino que hay una
        # amplia variabilidad en la columna. Los algoritmos posteriores se encargarán de despreciar dicha columna
        # de no servir para separar categorías

        datos[:, i] = columna

     #repite la etiqueta de cada parámetro cantCol de veces

    etiquetas_modificadas = []
    for etiqueta, n in zip(etiquetas, cantCol):
        if n > 1:
            for i in range(1, n + 1):
                etiqueta_modificada = f"{etiqueta}_{i}"
                etiquetas_modificadas.append(etiqueta_modificada)
        else:
            etiquetas_modificadas.append(etiqueta)

    etiquetas = np.array(etiquetas_modificadas)

    return datos, etiquetas

def clasificadores(xtrain, xtest, ytrain, ytest, param):

    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    from sklearn import metrics

    # Definir los posibles valores de hiperparámetros para cada modelo
    knn_parametros = {'n_neighbors': [3, 5, 7, 9, 11],
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

    svm_parametros = {'C': [0.1, 1, 10, 100],
                      'gamma': [0.01, 0.1, 1],
                      'kernel': ['linear', 'rbf', 'poly'],
                      'degree': [2, 3, 4]}

    rf_parametros = {'n_estimators': [50, 100, 150],
                     'criterion': ['gini', 'entropy'],
                     'max_depth': [2, 4, 6, 8],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4],
                     'max_features': ['sqrt', 'log2']}

    dt_parametros = {'criterion': ['gini', 'entropy'],
                     'splitter': ['best', 'random'],
                     'max_depth': [2, 4, 6, 8],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4],
                     'max_features': ['sqrt', 'log2']}

    # Lista de modelos y sus respectivos hiperparámetros
    modelos = [
        ('KNN', KNeighborsClassifier(), knn_parametros),
        ('SVM', SVC(), svm_parametros),
        ('RF', RandomForestClassifier(), rf_parametros),
        ('DT', DecisionTreeClassifier(),dt_parametros)
    ]

    resultados = {}
    # Preguntar al usuario si desea graficar el árbol de decisión
    respuesta = input('¿Desea graficar el árbol de decisión? (s/n): ')

    print("---------------------------------")
    print("Entrenando clasificadores...")

    for nombre, modelo, parametros in modelos:
        # Realizar la búsqueda en cuadrícula utilizando validación cruzada
        grid_search = GridSearchCV(modelo, parametros, cv=5) #verbose = 3 ; da metricas por fold
        grid_search.fit(xtrain, ytrain)

        # Obtener los mejores hiperparámetros y el mejor score
        mejores_parametros = grid_search.best_params_
        mejor_score = grid_search.best_score_

        # Ajustar el modelo con los mejores hiperparámetros
        modelo.set_params(**mejores_parametros)
        modelo.fit(xtrain, ytrain)

        # Calcular el rendimiento del modelo
        score_train = modelo.score(xtrain, ytrain)
        score_test = modelo.score(xtest, ytest)

        # Imprimir los resultados
        print('-------------------------------------')
        print('Resultados para el modelo:', nombre)
        print('Mejores hiperparámetros:', mejores_parametros)
        print('Mejor score:', mejor_score)
        print('Score en el conjunto de entrenamiento:', score_train)
        print('Score en el conjunto de prueba:', score_test)
        print('GE = 3;ME = 2; PE = 1; NE = 0;')


        if nombre == 'DT':

            if respuesta.lower() == 's':
                plt.figure(figsize=(10, 10))
                plot_tree(modelo, filled=True, feature_names=param, class_names=None)
                plt.show()

        y_pred = modelo.predict(xtest)
        print(metrics.classification_report(ytest, y_pred))

        # Almacenar los resultados en el diccionario
        resultados[nombre] = {
            'mejores_parametros': mejores_parametros,
            'mejor_score': mejor_score,
            'score_train': score_train,
            'score_test': score_test,
            'metricas_desempeño': metrics.classification_report(ytest, y_pred)
        }

    return resultados

main()