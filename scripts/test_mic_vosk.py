#for raspberry pi
# scripts/test_mic_vosk.py
import sys, json, numpy as np, sounddevice as sd, soundfile as sf
from vosk import Model, KaldiRecognizer
from pathlib import Path

MIC_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else None
MODEL_DIR = Path("models/vosk_model")
assert MODEL_DIR.exists(), f"Missing Vosk model at {MODEL_DIR} (unzip one here, e.g., 'vosk-model-small-en-us-0.15')"

def record(sec=8, sr=16000, ch=1, device=None):
    frames = int(sec * sr)
    audio = sd.rec(frames, samplerate=sr, channels=ch, dtype="int16", device=device)
    sd.wait()
    sf.write("debug_input.wav", audio, sr, subtype="PCM_16")
    return audio.tobytes(), sr

raw, sr = record(device=MIC_INDEX)
rec = KaldiRecognizer(Model(str(MODEL_DIR)), sr)
rec.AcceptWaveform(raw)
print(json.loads(rec.FinalResult()).get("text", "").strip())


#pip install vosk soundfile
#python -m scripts.test_mic_vosk 12
# or:
#python -m scripts.test_mic_vosk 16
