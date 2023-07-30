import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Carga de datos:
try:
    # Lee Parametros_Grandes y lo carga en un DataFrame
    df = pd.read_excel('D:\Tesis\Archivos Tesis FINAL\MuestrasAudio\Vigo Muestras\GrandeMuestras\Param.xlsx')

    # Selecciona una columnas específicas por sus nombres
    mfccMean_G = df['mfccMean']
    chromaMean_G = df['chromaMean']
    melSpecMean_G = df['melSpecMean']
    spectralContrastMean_G = df['spectralContrastMean']
    tonVect_G = df['tonVect']
    frec_max_G = df['frec_max']
    respFrec_G = df['respFrec_tercios']


    # Lee Parametros_Medianas y lo carga en un DataFrame
    df = pd.read_excel('D:\Tesis\Archivos Tesis FINAL\MuestrasAudio\Vigo Muestras\MedianaMuestras\Param.xlsx')

    mfccMean_M = df['mfccMean']
    chromaMean_M = df['chromaMean']
    melSpecMean_M = df['melSpecMean']
    spectralContrastMean_M = df['spectralContrastMean']
    tonVect_M = df['tonVect']
    frec_max_M = df['frec_max']
    respFrec_M = df['respFrec_tercios']

    # Lee Parametros_Pequeñas y lo carga en un DataFrame
    df = pd.read_excel('D:\Tesis\Archivos Tesis FINAL\MuestrasAudio\Vigo Muestras\PequeñaMuestras\Param.xlsx')

    mfccMean_P = df['mfccMean']
    chromaMean_P = df['chromaMean']
    melSpecMean_P = df['melSpecMean']
    spectralContrastMean_P = df['spectralContrastMean']
    tonVect_P = df['tonVect']
    frec_max_P = df['frec_max']
    respFrec_P = df['respFrec_tercios']

    # Lee Parametros_RuidoFondo y lo carga en un DataFrame
    df = pd.read_excel('D:/Tesis/Archivos Tesis FINAL/Parámetros/RF.xlsx')

    mfccMean_RF = df['mfccMean']
    chromaMean_RF = df['chromaMean']
    melSpecMean_RF = df['melSpecMean']
    spectralContrastMean_RF = df['spectralContrastMean']
    tonVect_RF = df['tonVect']
    frec_max_RF = df['frec_max']
    respFrec_RF = df['respFrec_tercios']

    # Convierte cada cadena en una lista de números
    mfccMean_G = np.stack(mfccMean_G.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    chromaMean_G = np.stack(chromaMean_G.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    melSpecMean_G = np.stack(melSpecMean_G.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    spectralContrastMean_G = np.stack(spectralContrastMean_G.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    tonVect_G = np.stack(tonVect_G.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    frec_max_G = np.stack(frec_max_G.apply(lambda x: np.fromstring(x[1:-1], sep=',')).values)
    respFrec_G = np.stack(respFrec_G.apply(lambda x: np.fromstring(x[1:-1], sep=',')).values)

    # Convierte cada cadena en una lista de números
    mfccMean_M = np.stack(mfccMean_M.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    chromaMean_M = np.stack(chromaMean_M.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    melSpecMean_M = np.stack(melSpecMean_M.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    spectralContrastMean_M = np.stack(spectralContrastMean_M.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    tonVect_M = np.stack(tonVect_M.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    frec_max_M = np.stack(frec_max_M.apply(lambda x: np.fromstring(x[1:-1], sep=',')).values)
    respFrec_M = np.stack(respFrec_M.apply(lambda x: np.fromstring(x[1:-1], sep=',')).values)

    # Convierte cada cadena en una lista de números
    mfccMean_P = np.stack(mfccMean_P.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    chromaMean_P = np.stack(chromaMean_P.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    melSpecMean_P = np.stack(melSpecMean_P.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    spectralContrastMean_P = np.stack(spectralContrastMean_P.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    tonVect_P = np.stack(tonVect_P.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    frec_max_P = np.stack(frec_max_P.apply(lambda x: np.fromstring(x[1:-1], sep=',')).values)
    respFrec_P = np.stack(respFrec_P.apply(lambda x: np.fromstring(x[1:-1], sep=',')).values)

    # Convierte cada cadena en una lista de números
    mfccMean_RF = np.stack(mfccMean_RF.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    chromaMean_RF = np.stack(chromaMean_RF.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    melSpecMean_RF = np.stack(melSpecMean_RF.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    spectralContrastMean_RF = np.stack(spectralContrastMean_RF.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    tonVect_RF = np.stack(tonVect_RF.apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values)
    frec_max_RF = np.stack(frec_max_RF.apply(lambda x: np.fromstring(x[1:-1], sep=',')).values)
    respFrec_RF = np.stack(respFrec_RF.apply(lambda x: np.fromstring(x[1:-1], sep=',')).values)

except:
    print('error al cargar de excels')

while True:
    # Preguntar por un input
    print('Parámetros disponibles: mfccMean , chromaMean , melSpecMean , spectralContrastMean, tonVect, frec_max, respFrec ')
    nuevo_input = input('Ingrese el parámetro que desea plotear (si es vacío termina el proceso): ')

    # Salir del loop si no se ingresa nada
    if nuevo_input == '':
        break

    grandes = nuevo_input + '_G'
    medianas = nuevo_input + '_M'
    pequeñas = nuevo_input + '_P'
    rfondo = nuevo_input + '_RF'


    #Cálculo de medias, medianas y desvíos para comparar por categoría
    #chromaMean

    media_G = np.mean(globals()[grandes], axis=0)
    mediana_G = np.median(globals()[grandes], axis=0)
    desvio_G = np.std(globals()[grandes], axis=0)

    media_M = np.mean(globals()[medianas], axis=0)
    mediana_M = np.median(globals()[medianas], axis=0)
    desvio_M = np.std(globals()[medianas], axis=0)

    media_P = np.mean(globals()[pequeñas], axis=0)
    mediana_P = np.median(globals()[pequeñas], axis=0)
    desvio_P = np.std(globals()[pequeñas], axis=0)

    media_RF = np.mean(globals()[rfondo], axis=0)
    mediana_RF = np.median(globals()[rfondo], axis=0)
    desvio_RF = np.std(globals()[rfondo], axis=0)

    num_etiquetas = len(media_G)
    x = list(range(1, num_etiquetas + 1))

    # Plot de las medias con barras de error para P, M y G
    #plt.errorbar(x, mediana_P, yerr= np.percentile(globals()[pequeñas], 75,axis=0) - np.percentile(globals()[pequeñas], 25,axis=0), label='P', fmt='-o', alpha=0.8)
    #plt.errorbar(x, mediana_M, yerr= np.percentile(globals()[medianas], 75,axis=0) - np.percentile(globals()[medianas], 25,axis=0), label='M', fmt='-o', alpha=0.8)
    #plt.errorbar(x, mediana_G, yerr= np.percentile(globals()[grandes] - np.percentile(globals()[grandes], 25,axis=0), 75,axis=0), label='G', fmt='-o', alpha=0.8)
    #plt.errorbar(x, mediana_RF, yerr=np.percentile(globals()[rfondo] - np.percentile(globals()[rfondo], 25,axis=0), 75,axis=0), label='RF', fmt='-o', alpha=0.8)

    plt.errorbar(x, media_P, yerr=desvio_P, label='P', fmt='-o', alpha=0.8)
    plt.errorbar(x, media_M, yerr=desvio_M, label='M', fmt='-o', alpha=0.8)
    plt.errorbar(x, media_G, yerr=desvio_G, label='G', fmt='-o', alpha=0.8)
    #plt.errorbar(x, mediana_RF, yerr=desvio_RF, label='RF', fmt='-o', alpha=0.8)

    # Leyenda
    plt.legend()

    # Generar una secuencia de números para las etiquetas en el eje x

    if nuevo_input == 'mfccMean' or nuevo_input == 'chromaMean' or nuevo_input == 'tonVect':
        # Etiquetas de las categorías en el eje x
        plt.xticks(x)  # Ubicaciones de los ticks en el eje x
        plt.xlabel('Coeficientes')

    if nuevo_input == "respFrec":
        plt.xticks(x,['5', '6.3', '8', '10', '12.5', '16', '20', '25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250', '315', '400', '500', '630', '800', '1000',
         '1250', '1600', '2000', '2500', '3150', '4000'], rotation=-90)
        plt.xlabel('Tercios de octava [Hz]')

    # plt.xlabel(x)  # Asignar las etiquetas generadas

    plt.show()


    #OTRO PLOT, descomentar si se quiere
    """
    colores = ['red', 'green', 'blue']
    plt.plot(x, media_P, color=colores[0], label='P')
    plt.fill_between(x, media_P - desvio_P, media_P + desvio_P, color=colores[0], alpha=0.2)
    plt.plot(x, media_M, color=colores[1], label='M')
    plt.fill_between(x, media_M - desvio_M, media_M + desvio_M, color=colores[1], alpha=0.2)
    plt.plot(x, media_G, color=colores[2], label='G')
    plt.fill_between(x, media_G - desvio_G, media_G + desvio_G, color=colores[2], alpha=0.2)
    plt.show()
    """
    # OTRO PLOT, descomentar si se quiere
    """
    # Convertir las matrices a tipo numérico
    matriz_grandes = globals()[grandes]
    matriz_bajas = globals()[pequeñas]
    matriz_medianas = globals()[medianas]
    matriz_rf = globals()[rfondo]

    # Configuración de los subplots
    fig, axs = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

    # Datos y colores correspondientes a cada categoría
    datos = [matriz_grandes, matriz_bajas, matriz_medianas, matriz_rf]
    colores = ['blue', 'green', 'red', 'orange']
    leyenda = ['Grandes', 'Bajas', 'Medianas', 'RF']

    # Crear un gráfico de cajas en cada subplot
    for i, ax in enumerate(axs):
        bp = ax.boxplot(datos[i], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colores[i])

        ax.set_xticklabels(['Banda {}'.format(j + 1) for j in range(len(datos[i][0]))])
        ax.set_title(leyenda[i])

    # Ajustar espacio entre subplots y etiquetas de los ejes
    plt.subplots_adjust(wspace=0.4)
    fig.text(0.5, 0.04, 'Banda', ha='center')
    fig.text(0.04, 0.5, 'Valor', va='center', rotation='vertical')

    # Mostrar los gráficos
    plt.show()
    """
