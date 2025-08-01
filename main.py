from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from matplotlib import pyplot as plt
from pydantic import BaseModel
import torchaudio
import torch
import subprocess
import pronouncing
import re
import whisper
import difflib
from jiwer import wer, cer
import noisereduce as nr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch.nn.functional as F
import io

app = FastAPI(title="PronuncIA")

# ——— Carga de modelos en startup() ———
@app.on_event("startup")
def load_models():
    global whisper_model, wav2vec_processor, wav2vec_model
    whisper_model = whisper.load_model("base")
    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_name)
    wav2vec_model.eval()

class TranscriptionResult(BaseModel):
    wav2vec_text: str
    whisper_text: str
    phonemes_espeak: str
    phonemes_arpabet: str
    wer: float
    cer: float

def obtener_fonemas_espeak(texto: str) -> str:
    res = subprocess.run(
        ['espeak-ng', '-q', '-x', texto],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return res.stdout.strip() if res.returncode == 0 else ""

def obtener_arpabet(texto: str) -> str:
    tokens = texto.lower().split()
    arp = []
    for w in tokens:
        phones = pronouncing.phones_for_word(w)
        arp.append(re.sub(r'\d', '', phones[0]) if phones else "[NA]")
    return ".".join(arp)

import tempfile
import os

@app.post(
    "/transcribe",
    response_model=TranscriptionResult,
    summary="Transcribe audio WAV"
)
async def transcribe_raw(
    wav: bytes = Body(
        ...,
        media_type="audio/wav",
        description="El archivo WAV debe ir como el cuerpo binario de la petición"
    )
):
    try:
        # ——— Cargamos el WAV desde memoria ———
        buffer = io.BytesIO(wav)
        signal, sr = torchaudio.load(buffer)

        # ——— Reducción de ruido ———
        denoised = nr.reduce_noise(y=signal[0].numpy(), sr=sr)
        signal = torch.from_numpy(denoised).unsqueeze(0)

        # ——— Resample si es necesario ———
        if sr != 16000:
            signal = torchaudio.transforms.Resample(sr, 16000)(signal)

        # ——— Wav2Vec2 ———
        inputs = wav2vec_processor(signal.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = wav2vec_model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        text_w2v = wav2vec_processor.batch_decode(ids)[0].lower().strip()

        # ——— Whisper usando fichero temporal ———
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav)
            tmp.flush()
            tmp_path = tmp.name

        res = whisper_model.transcribe(tmp_path, language="en")
        text_whisper = res["text"].strip()

        # Limpieza del fichero temporal
        os.remove(tmp_path)

        # ——— Fonemas y métricas ———
        ph_es_wav2 = obtener_fonemas_espeak(text_w2v)
        ph_ar_wav2 = obtener_arpabet(text_w2v)
        ph_es_whisper = obtener_fonemas_espeak(text_whisper)
        ph_ar_whisper = obtener_arpabet(text_whisper)
        met_wer = wer(text_w2v, text_whisper)
        met_cer = cer(text_w2v, text_whisper)

        # ——— Guardamos en estado para la gráfica si la necesitas ———
        app.state._last_logits = logits
        app.state._last_ids = ids

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar audio: {e}")

    return JSONResponse({
        "wav2vec_text": text_w2v,
        "whisper_text": text_whisper,
        "phonemes_espeak_wav2": ph_es_wav2,
        "phonemes_arpabet_wav2": ph_ar_wav2,
        "phonemes_espeak_whisper": ph_es_whisper,
        "phonemes_arpabet_whisper": ph_ar_whisper,
        "wer": met_wer,
        "cer": met_cer,
    })


def generar_confianza_png(logits, ids, processor) -> io.BytesIO:
    probs = F.softmax(logits, dim=-1).max(dim=-1).values.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(8, 2))
    plt.plot(probs)
    plt.xlabel("Frame")
    plt.ylabel("Confianza")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


@app.get("/")
def read_root():
    return {"hello": "world"}




# ——— Endpoint para la gráfica de confianza ———
@app.get("/confidence-plot")
def confidence_plot():
    if not hasattr(app.state, "_last_logits"):
        raise HTTPException(status_code=404, detail="No hay datos de última transcripción.")
    buf = generar_confianza_png(app.state._last_logits, app.state._last_ids, wav2vec_processor)
    return StreamingResponse(buf, media_type="image/png")
