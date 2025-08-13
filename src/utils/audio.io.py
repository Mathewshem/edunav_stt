# src/utils/audio_io.py
from io import BytesIO
from PIL import Image
import numpy as np, cv2, os, shutil
from pydub import AudioSegment
import soundfile as sf

import os, shutil
from pydub import AudioSegment

_ffmpeg = shutil.which("ffmpeg") or r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
_ffprobe = shutil.which("ffprobe") or r"C:\Program Files\ffmpeg\bin\ffprobe.exe"
if os.path.exists(_ffmpeg):  AudioSegment.converter = _ffmpeg
if os.path.exists(_ffprobe): AudioSegment.ffprobe = _ffprobe


# ----- tell pydub where ffmpeg lives (Windows) -----
_ffmpeg = shutil.which("ffmpeg") or r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
_ffprobe = shutil.which("ffprobe") or r"C:\Program Files\ffmpeg\bin\ffprobe.exe"
if os.path.exists(_ffmpeg):
    AudioSegment.converter = _ffmpeg
if os.path.exists(_ffprobe):
    AudioSegment.ffprobe = _ffprobe
# ----------------------------------------------------

def load_pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert("RGB")

def cv2_from_pil(img: Image.Image) -> "np.ndarray":
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def ensure_wav_pcm16(file_bytes: bytes) -> tuple[bytes, int]:
    """
    Convert arbitrary audio (mp3/m4a/ogg/wav/…) to WAV PCM16 mono 16k.
    Needs ffmpeg for formats like m4a/mp3.
    """
    # First try libsndfile (works for wav/flac/ogg)
    try:
        data, sr = sf.read(BytesIO(file_bytes), dtype='int16')
        buf = BytesIO()
        sf.write(buf, data, sr, subtype="PCM_16", format="WAV")
        return buf.getvalue(), sr
    except Exception:
        pass  # fall back to ffmpeg via pydub

    audio = AudioSegment.from_file(BytesIO(file_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    out = BytesIO()
    audio.export(out, format="wav")
    wav_bytes = out.getvalue()
    data, sr = sf.read(BytesIO(wav_bytes), dtype='int16')
    buf = BytesIO()
    sf.write(buf, data, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue(), sr
