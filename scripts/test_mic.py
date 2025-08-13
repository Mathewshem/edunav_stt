# scripts/test_mic.py
import sys, time
from src.engines.stt_google import transcribe_from_mic

mic_index = int(sys.argv[1]) if len(sys.argv) > 1 else None
lang = sys.argv[2] if len(sys.argv) > 2 else "en-KE"
sr = int(sys.argv[3]) if len(sys.argv) > 3 else 48000
ch = int(sys.argv[4]) if len(sys.argv) > 4 else 2
duration = 8

print(f"Using device: {mic_index} | language: {lang} | sr={sr} | ch={ch}")
print(f"Recording {duration}s — SPEAK NOW …")
for i in range(duration, 0, -1):
    print(f"{i}…", end="", flush=True); time.sleep(1)
print("\nTranscribing…")

text = transcribe_from_mic(duration_sec=duration, device=mic_index, language=lang,
                           sample_rate=sr, channels=ch)
print("Heard:", text)
print("Saved raw audio to debug_input.wav")
