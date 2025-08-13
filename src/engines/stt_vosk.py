#(optional, offline on Pi)
# pip install vosk
import json, io
import soundfile as sf
import numpy as np
from vosk import Model, KaldiRecognizer

_model_cache = {}

def _load_model(model_dir):
    key = str(model_dir)
    if key not in _model_cache:
        _model_cache[key] = Model(model_dir)
    return _model_cache[key]

def transcribe_wav_pcm16(wav_bytes: bytes, model_dir) -> str:
    model = _load_model(model_dir)
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype='int16')
    rec = KaldiRecognizer(model, sr)
    # stream in chunks
    chunk = 4000
    for i in range(0, len(data), chunk):
        rec.AcceptWaveform(data[i:i+chunk].tobytes())
    res = json.loads(rec.FinalResult())
    return (res.get("text") or "").strip()
