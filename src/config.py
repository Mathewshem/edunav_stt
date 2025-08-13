import os

STT_ENGINE = os.getenv("STT_ENGINE", "google").lower()
LANGUAGE = os.getenv("LANGUAGE", "en-KE")

def _get_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

MIC_DEVICE = _get_int("MIC_DEVICE", None)   # None means default input
MIC_SR     = _get_int("MIC_SR", 0)          # 0 means auto
MIC_CH     = _get_int("MIC_CH", 0)          # 0 means auto
