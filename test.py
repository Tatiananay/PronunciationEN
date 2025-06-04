import whisper
from phonemizer import phonemize


model = whisper.load_model("base")  # Usa "small", "medium" si tienes recursos
result = model.transcribe("audioTest/84-121123-0001.flac")
transcript = result["text"]
phonemes = phonemize(transcript, language='en-us', backend='espeak', strip=True)

print("Transcripción:", transcript)
print("Fonemas canónicos:", phonemes)