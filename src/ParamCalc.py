import librosa
import math
from os import listdir
from os.path import isfile, join
import soundfile as sf
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks
import PyOctaveBand as Bf
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure

def main():
    # Lee nombre de archivos que se encuentren en mypath

    mypath = "D:\Tesis\Archivos Tesis FINAL\MuestrasAudio\PruebasParaDetector\Muestras"

    # Calcula parámetros acústicos de archivos de audio

    listAudioData = paramCalc(mypath)

    # Exporta parámetros a excel

    df = pd.DataFrame(listAudioData)
    df.to_excel(mypath+'/Param.xlsx', index=False)

    return

def paramCalc(mypath,fsMin=11025):
    # Funcion encargada de calcular diversos parámetros acústicos.  11025
    """
    Funcion encargada de calcular diversos parámetros acústicos.

    Parametros de entrada:
    mypath -> dirección donde se encuentran los archivos a procesar
    """
    nomarch = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    cantArch = len(nomarch)

    k = 0
    print('Procesando...')
    # Procesa cada archivo por separado

    listAudioData = []

    for i in nomarch:
        #print(i)
        k = k + 1
        try:
            # Realiza el procesamiento de cada archivo y almacena en listas posteriores
            wavName = (mypath + '/' + i)
            nomArch = i
            x, fs = sf.read(wavName)

            # Si tiene una frecuencia de muestreo mayor, resamplea a fsMin (fs minima presente en las muestras)
            if fs != fsMin:
                x = librosa.resample(x, fs, fsMin)
                fs = fsMin

            try:
                #                       --- CALCULA PARAMETROS ---
                win_length = 2048
                hop_length = 512

                # Calculamos la respuesta en frecuencia en tercios de octava normalizados
                respFrec_tercios,freqs_tercios = terciosDeOctava(audio=x,fs=fs,plot=0)

                # Firma acústica 2 (información energética entre fmin y fmax)
                #band_starts, band_powers, pFrecs = firmAcoustic(audio=x, fs=fs, fmin=20, fmax=200, bandwidth=2, plot=0)

                # Índice de tonalidad:
                indTon,medPkFrec = tonCalc(x, fs, plot=0)

                # Índices de tonalidad (vector de índice de tonalidad calculado en subrangos de frecuencias)
                tonVect, frecuencias_maximas = tonVectCalc(x, fs, freq_max=200, freq_min=20, plot=0)

                #Contorno Espectral:
                #evolucionFrec,evolucionAmp = contornoEspectral(audio_data=x,sample_rate=fs, plot=0)
                #evolucionAmp = evolucionAmp/np.max(evolucionAmp) #normaliza con máximo para tener relación de niveles al pico máximo detectado en lugar de valores absolutos

                # SpectralCentroid
                spectralCentroid = librosa.feature.spectral_centroid(y=x, sr=fs, n_fft=win_length,
                                                                     hop_length=hop_length)
                spectralCentroid = spectralCentroid.squeeze(0)

                # spectral_contrast
                spectralContrast = librosa.feature.spectral_contrast(y=x, sr=fs, n_fft=win_length, hop_length=hop_length,fmin=50, n_bands=6)

                # spectral_bandwidth
                spectralBW = librosa.feature.spectral_bandwidth(y=x, sr=fs, n_fft=win_length, hop_length=hop_length)
                spectralBW = spectralBW.squeeze(0)

                # spectral_flatness
                spectralFlatness = librosa.feature.spectral_flatness(y=x, n_fft=win_length, hop_length=hop_length)
                spectralFlatness = spectralFlatness.squeeze(0)

                # Chromagram
                chroma = librosa.feature.chroma_stft(y=x, sr=fs, n_fft=win_length, hop_length=hop_length)

                # MFCC - Mel-frequency cepstral coefficient
                mfcc = librosa.feature.mfcc(y=x, sr=fs, n_fft=win_length, hop_length=hop_length)

                melSpec = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=win_length, hop_length=hop_length)

                # Fundamental estimation

                fundamentalYin = librosa.yin(y=x, fmin=5, fmax=4000, sr=fs, frame_length=2048, win_length=None,
                                          hop_length=hop_length,
                                          trough_threshold=0.1, center=True, pad_mode='constant')
                #fundamentalYin = [z for z in fundamentalYin if z <= 100] #elimina outliers (resultados superiores a 100Hz)

                # Spectral flux
                onset_env = librosa.onset.onset_strength(y=x, sr=fs)

                # ZC
                zeroCross = librosa.feature.zero_crossing_rate(x)
                zeroCross = zeroCross[0, :]

                # Inicializa objeto con parámetros
                featuresData = np.array([])

                # El siguiente bloque de codigo obtiene los valores medios de cada parámetro

                mfccMean = []
                for i in mfcc:
                    mfccMean.append(np.median(i))
                mfccMean = np.asarray(mfccMean)

                chromaMean = []
                for i in chroma:
                    chromaMean.append(np.median(i))
                chromaMean = np.asarray(chromaMean)

                melSpecMean = []
                for i in melSpec:
                    melSpecMean.append(20 * np.log10(np.median(i)))
                melSpecMean = np.asarray(melSpecMean)

                spectralContrastMean = []
                for i in spectralContrast:
                    spectralContrastMean.append(np.median(i))
                spectralContrastMean = np.asarray(spectralContrastMean)

                spectralCentroid = np.median(spectralCentroid)
                fundamentalYin = np.median(fundamentalYin)
                onset_env = np.median(onset_env)
                spectralFlatness = np.median(spectralFlatness)
                spectralBW = np.median(spectralBW)
                zeroCross = np.median(zeroCross)

                parametros = {} #diccionario con parámetros calculados
                parametros['nomArch'] = nomArch
                parametros['spectralCentroid'] = spectralCentroid
                parametros['spectralFlatness'] = spectralFlatness
                parametros['spectralBW'] = spectralBW
                parametros['onset_env'] = onset_env
                parametros['zeroCross'] = zeroCross
                parametros['mfccMean'] = mfccMean
                #parametros['chromaMean'] = chromaMean
                parametros['melSpecMean'] = melSpecMean
                parametros['spectralContrastMean'] = spectralContrastMean
                parametros['fundamentalYin'] = fundamentalYin

                #parametros['band_starts'] = np.asarray(band_starts)
                #parametros['band_powers'] = np.asarray(band_powers)
                #parametros['pFrecs'] = np.asarray(pFrecs)

                parametros['indTon'] = np.asarray(indTon)
                #parametros['medPkFrec'] = np.asarray(medPkFrec)

                parametros['tonVect'] = tonVect
                #parametros['frec_max'] = frecuencias_maximas

                #parametros['evolucionAmp'] = evolucionAmp
                #parametros['evolucionFrec'] = evolucionFrec

                parametros['respFrec_tercios'] = respFrec_tercios

                listAudioData.append(parametros)

            except:
                print('el archivo ' + i + ' dio error al calcular parámetros acústicos')

            print('Number of analized files: ' + str(k) + '/' + str(cantArch))

        except:
            print('File ' + str(i) + ' could not be loaded')
            print('Number of analized files: ' + str(k) + '/' + str(cantArch))

    return listAudioData

def tonCalc(audio, fs, plot=1):
    """
    La función tonCalc calcula el "índice de tonalidad" de una señal de audio. Primero, se calcula el espectro de amplitud
    de la señal y se aplica un filtro de promedio móvil para obtener una envolvente espectral. Luego, se detectan y
    filtran los picos significativos en el espectro. Opcionalmente, se muestra un gráfico que visualiza los resultados.

    :param audio: señal de audio
    :param fs: frecuencia de muestreo
    :param plot: habilita o no una representación gráfica (1 sí, 0 no)
    :return indTon, medPkFrec: índice de tonalidad, frecuencia media de los picos armónicos
    """

    # Calcular la Transformada de Fourier de la señal de audio
    fft_audio = np.abs(np.fft.rfft(audio,axis=0))

    # Calcular el espectro de amplitud
    amplitude_spectrum = 20 * np.log10(fft_audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / fs)

    # Define el tamaño de la ventana del filtro de promedio móvil
    window_size = int(np.round(len(amplitude_spectrum)/100))

    # Aplica el filtro de promedio móvil a las amplitudes de la señal
    smooth_magnitude = np.convolve(amplitude_spectrum, np.ones((window_size,)) / window_size, mode='same') #genera una especie de envolvente espectral para definir el umbral de deteccion de picos

    # Define los límites inferior y superior de rango de frecuencia a analizar
    freq_min = 30
    freq_max = 3000

    # Crea una máscara booleana para seleccionar solo las frecuencias dentro de ese rango
    freq_mask = np.logical_and(freqs >= freq_min, freqs <= freq_max)

    freqs=freqs[freq_mask]
    amplitude_spectrum=amplitude_spectrum[freq_mask]
    smooth_magnitude= smooth_magnitude[freq_mask]

    # Detecta los picos de la respuesta en frecuencia
    peaks, _ = find_peaks(amplitude_spectrum, height= 15+smooth_magnitude)

    # Descarta los picos que estén demasiado cerca
    freq_peaks = freqs[peaks]
    amps_peaks = amplitude_spectrum[peaks]
    # Calculamos la distancia proporcional a la frecuencia del pico (+- 10% de la frecuencia del pico)
    distance = 0.1 * freq_peaks

    # Descartamos los picos mas chicos que se encuentran dentro de la distancia mínima establecida
    for i in range(len(freq_peaks)):
        for j in range(i+1, len(freq_peaks)):
            if abs(freq_peaks[i] - freq_peaks[j]) <= distance[i]:
                if amps_peaks[i] < amps_peaks[j]:
                    peaks[i] = False
                else:
                    peaks[j] = False

    # arma los vectores con frecuencias y amplitudes de los picos resultantes (eliminando cuando peaks = 0)
    freq_harmonics = np.asarray([freqs[peaks[i]] for i in range(len(peaks)) if peaks[i] != 0])
    amp_harmonics = np.asarray([amplitude_spectrum[peaks[i]] for i in range(len(peaks)) if peaks[i] != 0])
    # obtiene el indice de tonalidad a partir de la modificación de la función sigmoide (para que de entre 0 y 1)

    indTon = (1 / (1 + math.exp(-(len(freq_harmonics)/ len(freqs))*12000))-0.5)*2

    medPkFrec = np.median(freq_harmonics)

    if plot == 1:
        # Traza la magnitud de la señal en función de las frecuencias dentro del rango especificado

        plt.figure(1)
        plt.clf()
        plt.semilogx(freqs, smooth_magnitude+15)
        plt.plot(freq_harmonics, amp_harmonics, "x")
        plt.plot(freqs, amplitude_spectrum,'r')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud')
        plt.show()

    return indTon, medPkFrec

def tonVectCalc(audio, fs,freq_max = 200,freq_min = 20, plot=1):

    """
    Esta funcion calcula el índice de tonalidad para cada bin de frecuencia, considerando la estabilidad temporal de la
    energíay la diferencia de energía con frecuencias adyacentes. Se obtienen las frecuencias máximas y los índices de
    tonalidad máximos en subrangos de frecuencia. Finalmente, se aplica una transformación sigmoide modificada para
    ajustar los valores de los índices de tonalidad.

    audio, fs = vector con informacion del audio a analizar y frecuencia de muestreo
    freq_max = 200 Frecuencia máxima a analizar
    freq_min = 20 Frecuencia mínima a analizar
    plot=1 Parámetro para decidir si representar gráficamente o no (1 sí, 0 no)

    El resultado es el vector de índices de tonalidad (tonVect) y las frecuencias asociadas (frecuencias_maximas)"""

    # Parámetros del espectrograma
    duration = len(audio) / fs
    nperseg = int(min(10 * duration * fs / freq_max, duration * fs * 0.1))
    noverlap = int(0.5 * nperseg)
    nfft = nperseg

    # Calcular el espectrograma
    freqs, times, spectrogram = signal.spectrogram(audio, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Encontrar el índice de frecuencia correspondiente a la banda de frecuencia de interés
    freq_idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]

    # Restringir las frecuencias al rango de interés
    freqs = freqs[freq_idx]
    spectrogram = spectrogram[freq_idx]

    # Inicializar el vector de correlación
    tonalVector = np.zeros(spectrogram.shape[0])

    # Calcular la tonalidad de cada bin de frecuencia a lo largo del tiempo
    for i in range(spectrogram.shape[0]):

        # Calcula el indice de tonalidad de cada bin teniendo en cuenta si hay un incremento energetico entre el bin en cuestion
        # y las bandas adyacentes (valores medio) y en función de que tan estatico el valor de la energia en el tiempo

        if i > 1 and i < spectrogram.shape[0] - 2:
            # calcula que tan constante en el tiempo es su energía (tiende a 1 si es contante o 0 si es aleatorio)
            tonal = np.sum((spectrogram[i] / np.max(spectrogram[i]))) / spectrogram.shape[1]

            # calcula el promedio energético entre +- 2 bins de frecuencia
            valMedio = (spectrogram[i - 2].mean() + spectrogram[i - 1].mean() + spectrogram[i].mean() + spectrogram[
                +11].mean() + spectrogram[i + 2].mean()) / 5
            # Obtiene el indice de tonalidad del i-esimo bin de frecuencia
            tonalVector[i] = (spectrogram[i].mean() / valMedio) * tonal
        else:
            tonalVector[i] = 0

    # Separo por subrangos de frecuencia de 20Hz de BW y me quedo con el valor de frecuencia donde tiene mayor valor de tonalidad
    tonMax = []
    frecuencias_maximas = []

    rango = 5
    inicio_rango = freq_min  # Frecuencia inicial del rango
    fin_rango = inicio_rango + rango  # Frecuencia final del rango

    while fin_rango <= freq_max:  # Frecuencia máxima permitida
        # Obtener subvector de índices de tonalidad dentro del rango
        indices_rango = tonalVector[(freqs >= inicio_rango) & (freqs < fin_rango)]

        # Obtener índice de tonalidad máximo y su frecuencia correspondiente
        indice_maximo = indices_rango.argmax()
        frecuencia_maxima = freqs[(freqs >= inicio_rango) & (freqs < fin_rango)][indice_maximo]

        frecuencias_maximas.append(frecuencia_maxima)

        # Almacena el índice de tonalidad máximo del subrango
        tonMax.append(tonalVector[(freqs >= inicio_rango) & (freqs < fin_rango)][indice_maximo])
        # Ajuste de rangos
        if inicio_rango > 50:
            rango = 10
        if inicio_rango > 80:
            rango = 20
        if inicio_rango > 120:
            rango = 40
        inicio_rango += rango
        fin_rango += rango

    #print(len(tonMax))
    # print(frecuencias_maximas)

    # modifica la relación lineal de los indices de tonalidad para que siga una forma sigmodie (modificada). Buscando tener
    # los indices superiores a 0.7 con valores cercanos a 1 e indices menores a 0.3 con valores cercanos a 0

    tonVect = 1 / (1 + np.exp(-(np.array(tonMax) * 10 - 5) * 1.2))

    if plot == 1:
        spectrogram = 20 * np.log10(spectrogram)

        plt.pcolormesh(times, freqs, spectrogram)
        plt.colorbar(label='dB')
        plt.title('Espectrograma')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')

        # Agregar líneas horizontales para los índices de tonalidad con color variable y transparencia
        for i in range(len(frecuencias_maximas)):
            color_intensity = tonVect[i]  # Intensidad de color en función del índice de tonalidad
            transparency = color_intensity  # Transparencia inversa al índice de tonalidad
            linestyle = '--'  # Estilo de línea punteada
            plt.axhline(y=frecuencias_maximas[i], color='r', alpha=transparency, linestyle=linestyle)

        plt.show()

    return tonVect, frecuencias_maximas

def terciosDeOctava(audio,fs,plot=1):

    amp, freqs = Bf.octavefilter(audio, fs, fraction=3, order=6, limits=[5, 4000], show=0, sigbands=0)
    freqs = [5, 6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
             2500, 3150, 4000]

    if plot == 1:
        # crear la figura y el eje
        fig, ax = plt.subplots()

        # crear un rango de índices para el eje x
        x_indexes = range(len(amp))
        # crear el gráfico de barras
        plt.bar(x_indexes, amp)
        # establecer los nombres de las etiquetas en el eje x
        plt.xticks(x_indexes, freqs)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud (dB)')
        # mostrar el gráfico
        plt.show()


    return amp, freqs

#estima la fundamental y los armonicos (en deshuso)
def fundArmCalc(audio, fs, plot=1):

    # Calcular la Transformada de Fourier de la señal de audio
    fft_audio = np.abs(np.fft.rfft(audio,axis=0))

    # Calcular el espectro de amplitud
    amplitude_spectrum = 20 * np.log10(fft_audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / fs)

    # Define el tamaño de la ventana del filtro de promedio móvil
    window_size = int(np.round(len(amplitude_spectrum)/500))

    # Aplica el filtro de promedio móvil a las amplitudes de la señal
    smooth_magnitude = np.convolve(amplitude_spectrum, np.ones((window_size,)) / window_size, mode='same') #genera una especie de envolvente espectral para definir el umbral de deteccion de picos

    # Define los límites inferior y superior de rango de frecuencia a analizar
    freq_min = 30
    freq_max = 3000

    # Crea una máscara booleana para seleccionar solo las frecuencias dentro de ese rango
    freq_mask = np.logical_and(freqs >= freq_min, freqs <= freq_max)

    freqs=freqs[freq_mask]
    amplitude_spectrum=amplitude_spectrum[freq_mask]
    smooth_magnitude= smooth_magnitude[freq_mask]

    # Detecta los picos de la respuesta en frecuencia
    peaks, _ = find_peaks(amplitude_spectrum, height= 15+smooth_magnitude)

    # Descarta los picos que estén demasiado cerca
    freq_peaks = freqs[peaks]
    amps_peaks = amplitude_spectrum[peaks]
    # Calculamos la distancia proporcional a la frecuencia del pico (+- 10% de la frecuencia del pico)
    distance = 0.1 * freq_peaks

    # Descartamos los picos mas chicos que se encuentran dentro de la distancia mínima establecida
    for i in range(len(freq_peaks)):
        for j in range(i+1, len(freq_peaks)):
            if abs(freq_peaks[i] - freq_peaks[j]) <= distance[i]:
                if amps_peaks[i] < amps_peaks[j]:
                    peaks[i] = False
                else:
                    peaks[j] = False

    # arma los vectores con frecuencias y amplitudes de los picos resultantes (eliminando cuando peaks = 0)
    freq_harmonics = np.asarray([freqs[peaks[i]] for i in range(len(peaks)) if peaks[i] != 0])
    amp_harmonics = np.asarray([amplitude_spectrum[peaks[i]] for i in range(len(peaks)) if peaks[i] != 0])

    # Convertir los valores a enteros multiplicándolos por un factor grande para reducir margen de error por convertir en entero
    factor = 10
    frequencies_int = np.round(freq_harmonics * factor).astype(int)

    # Elimina frecuencias para quedarse con las primeros 10 picos que detecta
    if len(freq_harmonics)>10:
        frequencies_int = frequencies_int[:10]
        amp_harmonics = amp_harmonics[:10]

    f1 = frequencies_int / frequencies_int[0]
    f2 = frequencies_int / frequencies_int[1]
    f3 = frequencies_int / frequencies_int[2]

    # reemplaza por 0 y 1 cada elemento de los arrays si el resultado es +-0.1 de de un valor entero y suma para ver cual de los 3 primeros picos tiene mas "potenciales" armonicos
    f1_score = np.sum(np.where(np.abs(np.mod(f1, 1)-0.5) < 0.4, 0, 1))
    f2_score = np.sum(np.where(np.abs(np.mod(f2, 1)-0.5) < 0.4, 0, 1))
    f3_score = np.sum(np.where(np.abs(np.mod(f3, 1)-0.5) < 0.4, 0, 1))
    max_score = max(f1_score, f2_score, f3_score)

    # Comprueba si hay más de un array fx_score con el valor máximo de scores
    # considera como fundamental al que tenga mayor amplitud

    if f1_score == f2_score == f3_score == max_score:
        # Si los tres arrays tienen el mismo score, devuelve el array con mayor amplitud
        max_amplitude_idx = np.argmax([amp_harmonics[0], amp_harmonics[1], amp_harmonics[2]])
        if max_amplitude_idx == 0:
            f_fund = f1
        elif max_amplitude_idx == 1:
            f_fund = f2
        else:
            f_fund = f3
    elif f1_score == f2_score == max_score:
        # Si los dos arrays tienen el mismo score, devuelve el array con mayor amplitud
        max_amplitude_idx = np.argmax([amp_harmonics[0], amp_harmonics[1]])
        if max_amplitude_idx == 0:
            f_fund = f1
        else:
            f_fund = f2
    elif f1_score == f3_score == max_score:
            # Si los dos arrays tienen el mismo score, devuelve el array con mayor amplitud
            max_amplitude_idx = np.argmax([amp_harmonics[0], amp_harmonics[2]])
            if max_amplitude_idx == 0:
                f_fund = f1
            else:
                f_fund = f3
    elif f2_score == f3_score == max_score:
            # Si los dos arrays tienen el mismo score, devuelve el array con mayor amplitud
            max_amplitude_idx = np.argmax([amp_harmonics[1], amp_harmonics[2]])
            if max_amplitude_idx == 0:
                f_fund = f2
            else:
                f_fund = f3
    else:
        #Si no son iguales los scores, se queda con el mayor
        max_index = np.argmax(np.array([f1_score, f2_score, f3_score]))  # obtiene cual fx_score es mayor
        # Define fundamental en funcion del fx_score máximo
        if max_index == 0:
            f_fund = f1
        elif max_index == 1:
            f_fund = f2
        else:
            f_fund = f3

    mask = np.where(np.abs(np.mod(f_fund, 1)-0.5) < 0.4, 0, 1)
    # Generar nuevo array con elementos que se consideran armónicos (donde la máscara vale 1)
    fundArm = frequencies_int[mask == 1]
    amp_harmonics = amp_harmonics[mask==1]

    # Convertir el valor de la frecuencia fundamental de nuevo a float
    fundArm = fundArm/factor

    # Asignar frecuencia y amplitud de fundamental y sus dos primeros armónicos (si pudo detectar)
    try:
        fund_freq = fundArm[0]
        fund_amp = amp_harmonics[0]
    except:
        fund_freq = 0
        fund_amp = 0

    try:
        fharm_freq = fundArm[1]
        fharm_amp = amp_harmonics[1]
    except:
        fharm_freq = 0
        fharm_amp = 0

    try:
        sharm_freq = fundArm[2]
        sharm_amp = amp_harmonics[2]
    except:
        sharm_freq = 0
        sharm_amp = 0


    if plot == 1:
        # Traza la magnitud de la señal en función de las frecuencias dentro del rango especificado (Descomentar para ver)
        #"""
        plt.figure(1)
        plt.clf()
        plt.semilogx(freqs, smooth_magnitude)
        plt.plot(fundArm, amp_harmonics, "x")
        plt.plot(freqs, amplitude_spectrum,'r')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud')
        plt.show()
        #"""
    freqs = [fund_freq,fharm_freq,sharm_freq]
    amps = [fund_amp,fharm_amp,sharm_amp]

    return freqs,amps

#saca un contorno con logica de deteccion de picos (en deshuso)
def contornoEspectral(audio_data, sample_rate, freq_min=20, freq_max=500, num_splits=1, cant_peaks=20, plot=1):
    """
    Funcion encargada de obtener el contorno espectral de un archivo de audio.

    Parametros de entrada:
    wavName -> nombre del archivo a procesar con path incluido
    freq_min -> Frecuencia mínima en Hz
    freq_max  -> Frecuencia máxima en Hz
    num_splits -> cantidad de subframes
    cant_peaks -> Cantidad de picos que detecta por subrfame
    """

    # Aplicar una ventana de Hamming para reducir el efecto de las discontinuidades en los extremos del audio
    window = np.hamming(len(audio_data))
    audio_data = audio_data * window

    # Definir los parámetros del espectrograma
    duration = len(audio_data) / sample_rate
    nperseg = int(min(10 * duration * sample_rate / freq_max, duration * sample_rate * 0.1))
    noverlap = int(0.5 * nperseg)
    nfft = nperseg

    # Aplicar la transformada de Fourier de corto tiempo (STFT) para obtener el espectrograma de frecuencia
    freqs, times, spectrogram = signal.spectrogram(audio_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
                                                   nfft=nfft)


    # Encontrar el índice de frecuencia correspondiente a la banda de frecuencia de interés
    freq_idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]

    # Restringir las frecuencias al rango de interés
    freqs = freqs[freq_idx]
    spectrogram = spectrogram[freq_idx]

    amplitudes = 20 * np.log10(spectrogram)

    # LOGICA PARA REALZAR TONOS PARA QUE SE DETECTE MEJOR!!!!! ------------------------------------------

    # Define el tamaño de la ventana del filtro de promedio móvil
    window_size = int(10)

    # Aplica el filtro de promedio móvil a las amplitudes de la señal
    suave = np.convolve(amplitudes.mean(axis=1), np.ones((window_size,)) / window_size, mode='same')
    suave2 = amplitudes.mean(axis=1)
    amplitudes = amplitudes + suave2.reshape(len(suave2),1)*2 - suave.reshape(len(suave),1)*0.5

    # calcular los valores promedios de energía de cada frecuencia
    specProm = np.mean(spectrogram, axis=1)
    specProm = 20 * np.log10(specProm)

    # Divide en num_splits el espectrograma para calcular en subframes
    spectrogram_splits = np.array_split(spectrogram, num_splits, axis=1)
    peaks = []

    evolucionFrec = []
    evolucionAmp = []

    for i in range(num_splits):

        # Encontrar los picos en el espectrograma

        specMean = spectrogram_splits[i].mean(axis=1)
        specMean = 20 * np.log10(specMean)
        specMax = spectrogram_splits[i].max(axis=1)

        # Define el tamaño de la ventana del filtro de promedio móvil
        window_size = int(50)

        # Aplica el filtro de promedio móvil a las amplitudes de la señal
        smooth_magnitude = np.convolve(specMean, np.ones((window_size,)) / window_size, mode='same')

        peaks, _ = find_peaks(specMean, height=smooth_magnitude)

        #print(peaks)

        freqs_peaks = freqs[peaks]
        amps_peaks = 20 * np.log10(specMean[peaks])
        # Calculamos la distancia proporcional a la frecuencia del pico (+- 10% de la frecuencia del pico (ej 50hz +- 5hz)
        distance = 0.1 * freqs_peaks

        # Descartamos los picos mas chicos que se encuentran dentro de la distancia mínima establecida
        for i in range(len(freqs_peaks)):
            for j in range(i + 1, len(freqs_peaks)):
                if abs(freqs_peaks[i] - freqs_peaks[j]) <= distance[i]:
                    if amps_peaks[i] < amps_peaks[j]:
                        peaks[i] = False
                    else:
                        peaks[j] = False

        # Obtener las frecuencias y amplitudes de los picos
        peak_frequencies = freqs[peaks]
        peak_amplitudes = 20 * np.log10(spectrogram.max(axis=1)[peaks])

        # Ordenar los picos por nivel
        sorted_peak_indices = sorted(range(len(peak_amplitudes)), key=lambda k: peak_amplitudes[k], reverse=True)

        if len(peaks) < 5: #si encuentra menos picos de que superen x veces la envolvente frecuencial, toma los picos maximos absolutos hasta completar los cant_peaks requeridos
            peaks, _ = find_peaks(specMax, height=0)

            freqs_peaks = freqs[peaks]
            amps_peaks = 20 * np.log10(spectrogram.max(axis=1)[peaks])
            # Calculamos la distancia proporcional a la frecuencia del pico (+- 10% de la frecuencia del pico (ej 50hz +- 5hz)
            distance = 0.1 * freqs_peaks

            # Descartamos los picos mas chicos que se encuentran dentro de la distancia mínima establecida
            for i in range(len(freqs_peaks)):
                for j in range(i + 1, len(freqs_peaks)):
                    if abs(freqs_peaks[i] - freqs_peaks[j]) <= distance[i]:
                        if amps_peaks[i] < amps_peaks[j]:
                            peaks[i] = False
                        else:
                            peaks[j] = False

            # Obtener las frecuencias y amplitudes de los picos
            peak_frequencies = freqs[peaks]
            peak_amplitudes = 20 * np.log10(spectrogram.max(axis=1)[peaks])

            # Ordenar los picos por nivel
            sorted_peak_indices = sorted(range(len(peak_amplitudes)), key=lambda k: peak_amplitudes[k], reverse=True)



        # Seleccionar los 5 picos tonales más prominentes
        top_peak_indices = sorted_peak_indices[:cant_peaks]

        # Obtener las frecuencias y amplitudes de los 5 picos tonales más prominentes
        top_peak_frequencies = peak_frequencies[top_peak_indices]
        top_peak_amplitudes = peak_amplitudes[top_peak_indices]

        # Ordena los picos de min frec a max frec
        sorted_peak_indices = np.argsort(top_peak_frequencies)
        top_peak_frequencies = top_peak_frequencies[sorted_peak_indices]
        top_peak_amplitudes = top_peak_amplitudes[sorted_peak_indices]

        evolucionFrec.append(top_peak_frequencies)
        evolucionAmp.append(top_peak_amplitudes)

    # Trazar el espectrograma

    if plot == 1:
        plt.figure(2)
        plt.clf()
        plt.pcolormesh(times, freqs, amplitudes, cmap='jet')
        plt.ylabel('Frecuencia [Hz]')
        plt.xlabel('Tiempo [s]')
        plt.colorbar()

        """
        # trazar las líneas horizontales
        for i in range(num_splits):
            # obtener el índice de inicio y fin del tercio actual
            start_idx = int(i * len(times) / num_splits)
            end_idx = int((i + 1) * len(times) / num_splits) - 1

            # trazar las líneas rojas horizontales para los picos de este tercio
            for freq in zip(evolucionFrec[i]):
                x_vals = [times[start_idx], times[end_idx]]
                y_vals = [freq, freq]
                plt.plot(x_vals, y_vals, color='black', linestyle='--')

            # Graficar la línea vertical que separa el tercio actual
            if i < num_splits - 1:
                plt.axvline(times[end_idx], color='gray', linestyle='--')
        """
        plt.show()

    # Convertir las listas de arrays en un único array
    array = np.array(evolucionFrec)
    array = np.transpose(array)
    evolucionFrec = np.ravel(array)

    array = np.array(evolucionAmp)
    array = np.transpose(array)
    evolucionAmp = np.ravel(array)

    # Esto se hace para poder evaluar en cada detección por cuanto supera al nivel promedio. Se consulta
    # cada frecuencia donde detectó el pico (evolucionFrec) para comparar con el nivel promedio en esta frec en particular
    window_size = 5
    specProm_suavizado = np.convolve(specProm, np.ones(window_size) / window_size, mode='same')
    specProm_suavizado[0] = specProm[0]
    specProm_suavizado[1] = (specProm[0] + specProm[1] + specProm[2]) / 3
    specProm_suavizado[-1] = (specProm[-1])
    specProm_suavizado[-2] = (specProm[-1] + specProm[-2] + specProm[-3]) / 3
    # crea un diccionario donde key = freqs; value = specProm
    diccionario = dict(zip(freqs, specProm_suavizado))
    evolucionAmp2 = []
    for i in range(len( evolucionFrec)):
        frecuencia = evolucionFrec[i]
        amplitud = evolucionAmp[i]
        evolucionAmp2.append(amplitud - diccionario[frecuencia])

    evolucionAmp = np.asarray(evolucionAmp2)

    #print(evolucionFrec)

    return evolucionFrec,evolucionAmp

#saca una firma acustica (bandas de BW = badwidth) entre fmin y fmax (en deshuso)
def firmAcoustic(audio,fs, fmin= 20, fmax=100, bandwidth=2, plot=1):
    """
    Funcion encargada de obtener el contorno espectral de un archivo de audio.

    Parametros de entrada:
    wavName -> nombre del archivo a procesar con path incluido
    fmin -> Frecuencia mínima en Hz
    fmax  -> Frecuencia máxima en Hz
    bandwidth -> BW de cada banda
    cant_peaks -> Cantidad de picos que detecta por subrfame
    """

    # Calcular la transformada de Fourier de la señal de audio
    N = len(audio)
    fft = np.fft.fft(audio)

    # Calcular la magnitud de la transformada
    magnitude = np.abs(fft)

    # Calcular la potencia espectral
    power = (magnitude ** 2) / (N ** 2)

    # Calcular el espectro de potencia en dB
    power_db = 10 * np.log10(power)

    # Definir los límites de las bandas
    band_starts = np.arange(fmin, fmax, bandwidth)
    band_ends = band_starts + bandwidth

    # Calcular las medias de potencia en cada banda
    band_powers = []
    for i in range(len(band_starts)):
        start = int(band_starts[i] * N / fs)
        end = int(band_ends[i] * N / fs)
        band_powers.append(np.mean(power_db[start:end]))

    sorted_idxs = np.argsort(band_powers)[::-1]
    top_5_idxs = sorted_idxs[:5]
    top_5_powers = []
    top_5_freqs = []

    for idxs in top_5_idxs:
        top_5_powers.append(band_powers[idxs])
        top_5_freqs.append(band_starts[idxs])

    pFrecs = sorted(top_5_freqs, reverse=False)

    if plot == 1:
        # Graficar la potencia media en dB de cada banda
        plt.bar(band_starts, band_powers, width=bandwidth, align="edge")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.show()

    return band_starts, band_powers, pFrecs



    main()

main()

