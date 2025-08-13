# src/engines/stt_google.py
import io
import numpy as np
import sounddevice as sd
import speech_recognition as sr_mod  # alias to avoid name collision

LANG_FALLBACKS = ("en-KE", "en-US", "en-GB", "sw-KE")

def transcribe_from_file(file_bytes: bytes, language: str = "en-KE") -> str:
    # Lazy import so pydub/ffmpeg only loads if we actually handle a file
    from ..utils.audio_io import ensure_wav_pcm16
    wav_bytes, _ = ensure_wav_pcm16(file_bytes)
    r = sr_mod.Recognizer()
    with sr_mod.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language=language or LANG_FALLBACKS[0])
    except sr_mod.UnknownValueError:
        for lang in LANG_FALLBACKS:
            try:
                return r.recognize_google(audio, language=lang)
            except sr_mod.UnknownValueError:
                continue
        raw = r.recognize_google(audio, language=LANG_FALLBACKS[0], show_all=True)
        raise RuntimeError(f"No speech recognized from file. Raw={raw}")

def _to_mono_int16(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = arr.mean(axis=1)
    arr = np.clip(arr, -32768, 32767)
    return arr.astype(np.int16)

def _resample_to_16k_int16(arr: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz using simple decimation if possible, else linear interp."""
    if sr == 16000:
        return _to_mono_int16(arr)
    # flatten to mono first (float32)
    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = arr.mean(axis=1)
    x = arr.astype(np.float32)
    if sr % 16000 == 0:
        factor = sr // 16000
        y = x[::factor]
    else:
        # linear interpolation
        n_new = int(round(len(x) * 16000.0 / sr))
        xp = np.linspace(0, 1, num=len(x), endpoint=False, dtype=np.float32)
        fp = x
        x_new = np.linspace(0, 1, num=n_new, endpoint=False, dtype=np.float32)
        y = np.interp(x_new, xp, fp).astype(np.float32)
    y = np.clip(y, -32768, 32767)
    return y.astype(np.int16)

def transcribe_from_mic(
    duration_sec: int = 8,
    sample_rate: int | None = None,   # we will try forcing 16k internally
    channels: int | None = None,
    language: str = "en-KE",
    device: int | None = None,
    save_debug_wav: bool = True
) -> str:
    # Inspect device to learn its native sample rate
    dev_info = sd.query_devices(device, 'input')
    native_rate = int(dev_info.get('default_samplerate') or 48000)
    # We’ll try these (rate, channels) combos, in order:
    combos = [
        (16000, 1), (16000, 2),                 # ideal for ASR
        (native_rate, 1), (native_rate, 2),     # fallback to device native
        (48000, 1), (48000, 2),                 # common rates
    ]
    # If caller forced sr/ch, put it at the front
    if sample_rate or channels:
        combos.insert(0, (sample_rate or native_rate, channels or 1))

    recording = None
    used_rate = None
    last_err = None
    frames = None

    for dev_choice in (device, None):  # try selected device, then default input
        for rate, ch in combos:
            try:
                frames = int(duration_sec * rate)
                buf = sd.rec(frames, samplerate=rate, channels=ch,
                             dtype="int16", device=dev_choice)
                sd.wait()
                recording = buf
                used_rate = rate
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue
        if recording is not None:
            break

    if recording is None or used_rate is None:
        raise RuntimeError(f"Audio capture failed: {last_err}")

    # Downmix to mono and light auto-gain
    rec = _to_mono_int16(recording)
    rms = float(np.sqrt(np.mean((rec.astype(np.float32) / 32768.0) ** 2)) + 1e-12)
    if rms < 0.005:
        target = 0.1
        gain = min(12.0, target / max(rms, 1e-6))
        rec = np.clip(rec.astype(np.float32) * gain, -32768, 32767).astype(np.int16)

    # Resample to 16k (Google is happiest with this)
    rec16 = _resample_to_16k_int16(rec, used_rate)

    # Optional debug
    if save_debug_wav:
        import soundfile as sf
        sf.write("debug_input.wav", rec16, 16000, subtype="PCM_16")

    # Build AudioData@16k and recognize with fallbacks
    audio_data = sr_mod.AudioData(rec16.tobytes(), 16000, 2)
    r = sr_mod.Recognizer()
    try:
        return r.recognize_google(audio_data, language=language or LANG_FALLBACKS[0])
    except sr_mod.UnknownValueError:
        for lang in LANG_FALLBACKS:
            try:
                return r.recognize_google(audio_data, language=lang)
            except sr_mod.UnknownValueError:
                continue
        raw = r.recognize_google(audio_data, language=LANG_FALLBACKS[0], show_all=True)
        raise RuntimeError(f"No speech recognized from mic. Raw={raw}")
