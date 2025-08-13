#(optional; heavy on Pi 2 GB)
# pip install openai-whisper
import io, numpy as np, soundfile as sf, whisper

_model = None

def _lazy_load(model_size="base"):
    global _model
    if _model is None:
        _model = whisper.load_model(model_size)
    return _model

def transcribe_wav_pcm16(wav_bytes: bytes, model_size="base") -> str:
    model = _lazy_load(model_size)
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32')
    if sr != 16000:
        # whisper prefers 16k; resample via numpy (simple)
        import resampy
        audio = resampy.resample(audio, sr, 16000)
    result = model.transcribe(audio, language="en")
    return (result.get("text") or "").strip()
