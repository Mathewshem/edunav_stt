import io
from pydub import AudioSegment
import soundfile as sf
import numpy as np

def ensure_wav_pcm16(file_bytes: bytes) -> tuple[bytes, int]:
    """
    Convert arbitrary audio (mp3/m4a/ogg/wav) to WAV PCM16 mono 16kHz.
    Returns (wav_bytes, sample_rate)
    """
    # Load via pydub (ffmpeg backend)
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    out_buf = io.BytesIO()
    audio.export(out_buf, format="wav")
    wav_bytes = out_buf.getvalue()
    # peek sr from exported buffer for downstream libs needing ndarray
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype='int16')
    # re-encode to ensure PCM16 little-endian contiguous
    buf = io.BytesIO()
    sf.write(buf, data.astype(np.int16), sr, subtype="PCM_16", format="WAV")
    return buf.getvalue(), sr
