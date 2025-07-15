from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio #MAnejo de audio con PyToorch
import torch
import subprocess #Comandos para espeak
import sounddevice as sd #Grabar audio
from scipy.io.wavfile import write 
import pronouncing
import re
import whisper
import difflib
from jiwer import wer, cer
from matplotlib import pyplot as plt
import torch.nn.functional as F
import noisereduce as nr #PAra eliminar ruido del audio
import keyboard  # Agregado para detectar teclas
import numpy as np  # Agregado para concatenar arrays
from scipy.signal import butter, lfilter

# ---------- CONFIGURACION ----------

DURACION = 5    
ARCHIVO_SALIDA = "grabacion.wav"

# ---------- GRABAR AUDIO ----------

def grabar_audio(duracion, archivo_salida):
    try:
        print(f"\nIniciando grabación: {duracion} segundos...")
        #Teorema de Nyquist-Shannon: la frecuencia de muestreo debe ser al menos el doble de la frecuencia máxima del sonido a grabar
        frecuencia_muestreo = 18000 #Recomendado para voz 16
        audio = sd.rec(int(duracion * frecuencia_muestreo), #Numero total de muestras por gabrar  
                       samplerate=frecuencia_muestreo, #Numero de muestras por segundo
                       channels=1, 
                       dtype='int16') #Permite capturar suficientes detalles sin ocupar demasiado espacio
        sd.wait() 
        write(archivo_salida, frecuencia_muestreo, audio)
        print("Grabación realizada.")
    except Exception as error:
        print(f"Error al grabar audio: {error}")
        return False
    return True
def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def aplicar_bandpass(data, fs):
    b, a = butter_bandpass(300, 3400, fs)
    y = lfilter(b, a, data)
    return y

# -------- NORMALIZAR AUDIO --------

def normalizar_audio(audio):
    max_valor = np.max(np.abs(audio))
    if max_valor == 0:
        return audio
    return audio / max_valor

#---------- GRABAR AUDIO EN TIEMPO REAL (experimental) ----------
def grabar_audio_tiempo_real(nombre_archivo="grabacion.wav"):
    try:
        print("Presione 's' para detener la grabación.")
        frecuencia_muestreo = 16000
        audio_total = []

        def callback(indata, frames, time, status):
            audio_total.append(indata.copy())

        with sd.InputStream(samplerate=frecuencia_muestreo, channels=1, dtype='float32', callback=callback):
            while not keyboard.is_pressed('s'):
                sd.sleep(100)  # verifica cada 0.1 seg si presionaron 's'

        # Concatenar bloques grabados
        audio_np = np.concatenate(audio_total, axis=0).flatten()

        # Reducción de ruido
        print("Reduciendo ruido...")
        audio_sin_ruido = nr.reduce_noise(y=audio_np, sr=frecuencia_muestreo)

        # Normalización
        audio_normalizado = normalizar_audio(audio_sin_ruido)

        # Filtrado de frecuencias
        audio_filtrado = aplicar_bandpass(audio_normalizado, frecuencia_muestreo)

        # Convertir a int16 para guardar WAV
        audio_int16 = np.int16(audio_filtrado * 32767)
        write(nombre_archivo, frecuencia_muestreo, audio_int16)

        print("Grabación mejorada guardada.")
        return audio_int16, frecuencia_muestreo

    except Exception as e:
        print(f"Error en la grabación mejorada: {e}")
        return None, None

# ---------- OBTENER FONEMAS (espeak-ng) ----------

#Espeak convierte de texto a fonemas (Se puede borrar xd)
# -q es para que no hable, -x es para que devuelva los fonemas
def obtener_fonemas(texto):
    try:
        resultado = subprocess.run(['espeak-ng', '-q', '-x', texto], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return resultado.stdout.strip() #Se limpia el resultado de espacios de inicio al final
    except Exception as error:
        return None
    
#---------- OBTENER ARPAbet (CMUdict) ----------
def obtener_arpabet(texto):
    palabras = texto.lower().split() # divide el texto en palabras y lo convierte en minusculas
    transcripcion_arpabet = []
    for palabra in palabras:
        fonemas = pronouncing.phones_for_word(palabra) # devuelve fonemas ARPAbet 
        if fonemas:
            sin_estres = re.sub(r'\d', '', fonemas[0]) 
            transcripcion_arpabet.append(sin_estres)
        else:
            transcripcion_arpabet.append("[NA]")
    return '.'.join(transcripcion_arpabet)

# ---------- CARGAR MODELOS ----------

def cargar_modelos():
    print("Cargando modelos...")
    modelo_whisper = whisper.load_model("base") 

    nombre_modelo = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    procesador = Wav2Vec2Processor.from_pretrained(nombre_modelo)
    modelo_wav2vec = Wav2Vec2ForCTC.from_pretrained(nombre_modelo)

    return modelo_whisper, procesador, modelo_wav2vec

# ---------- TRANSCRIBIR CON WHISPER ----------

def transcribir_con_whisper(ruta_audio):
    modelo = whisper.load_model("base")
    resultado = modelo.transcribe(ruta_audio, language='en')
    return resultado['text']

# ---------- DIFERENCIAS ----------

def resaltar_diferencias(esperado, actual):
    palabras_esperadas = esperado.lower().split()
    palabras_actuales = actual.lower().split()

    diferencias = difflib.ndiff(palabras_esperadas, palabras_actuales)
    print("\nDiferencias:")
    for d in diferencias:
        if d.startswith("- "):
            print("Palabras perdidas o incorrectas:", d[2:])
        elif d.startswith("+ "):
            print("Palabra de más:", d[2:])

# ---------- COMPARACION DETALLADA ----------

#usa jiwer
def comparacion_detallada(esperado, actual):
    print(f"\nAnálisis detallado:")
    print(f"Dicho: {esperado}")
    print(f"Actual:   {actual}")
    print(f"Tasa de error por palabra (WER): {wer(esperado, actual):.2%}")
    print(f"Tasa de error por caracter (CER): {cer(esperado, actual):.2%}")

    palabras_esperadas = esperado.lower().split()
    palabras_actuales = actual.lower().split()

#-----------GRAFICAR RUIDO ----------
def graficar_confianza(logits, ids_predichos, procesador):
    probabilidades = F.softmax(logits, dim=-1)
    max_probabilidades = probabilidades.max(dim=-1).values.squeeze().detach().cpu().numpy()
    tokens = procesador.batch_decode(ids_predichos)[0]

    plt.figure(figsize=(12, 4))
    plt.plot(max_probabilidades, label="Confianza")
    plt.xlabel("Frame de tiempo")
    plt.ylabel("Confianza (0-1)")
    plt.title("Nivel de confianza por fragmento de audio")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

# ---------- MAIN ----------

if __name__ == "__main__":

    audio_data, frecuencia = grabar_audio_tiempo_real()
    if audio_data is None:
        print("Error en la grabación continua")
        exit()
        
    # Guardar el audio continuo
    write(ARCHIVO_SALIDA, frecuencia, audio_data)
    signal = torch.tensor(audio_data.flatten(), dtype=torch.float32).unsqueeze(0)
    if frecuencia != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=frecuencia, new_freq=16000)(signal)

    # Cargar modelos
    modelo_whisper, procesador, modelo_wav2vec = cargar_modelos()

    # Transcripción con wav2vec2
    print("Transcribiendo")
    
    entradas = procesador(signal[0], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = modelo_wav2vec(**entradas).logits
    ids_predichos = torch.argmax(logits, dim=-1)
    texto_transcrito = procesador.batch_decode(ids_predichos)[0]

    print("\nTranscripción:", texto_transcrito)
    fonemas = obtener_fonemas(texto_transcrito.lower())
    print("Fonemas (espeak-ng):", fonemas)
    print()
    arpabet = obtener_arpabet(texto_transcrito)
    print("Fonemas ARPAbet (CMUdict):", arpabet)
    print()

    # Transcripción con Whisper
    transcripcion_whisper = transcribir_con_whisper(ARCHIVO_SALIDA)
    print("\nTranscripción con Whisper:", transcripcion_whisper)

    comparacion_detallada(texto_transcrito, transcripcion_whisper)
    
    logits = modelo_wav2vec(**entradas).logits
    ids_predichos = torch.argmax(logits, dim=-1)

    graficar_confianza(logits, ids_predichos, procesador)
