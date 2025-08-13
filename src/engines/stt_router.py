from ..config import STT_ENGINE

def transcribe_file(file_bytes: bytes, language: str):
    if STT_ENGINE == "vosk":
        from . import stt_vosk  # optional module (offline)
        from ..utils.audio_io import ensure_wav_pcm16
        wav_bytes, _ = ensure_wav_pcm16(file_bytes)
        text = stt_vosk.transcribe_wav_pcm16(wav_bytes, model_dir=None)  # stt_vosk reads its default path
        return text, "vosk"
    else:
        from . import stt_google
        text = stt_google.transcribe_from_file(file_bytes, language=language)
        return text, "google"

def transcribe_mic(duration_sec: int, language: str, sample_rate: int | None, channels: int | None, device: int | None):
    if STT_ENGINE == "vosk":
        # For Vosk, prefer the /stt (file) path, or wire a mic capture just like google and feed to vosk
        raise NotImplementedError("Mic path is only implemented for google in this service. Use /stt with Vosk.")
    else:
        from . import stt_google
        text = stt_google.transcribe_from_mic(
            duration_sec=duration_sec,
            language=language,
            sample_rate=sample_rate if sample_rate else None,
            channels=channels if channels else None,
            device=device
        )
        return text, "google"
