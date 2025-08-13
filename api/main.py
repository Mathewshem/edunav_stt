import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from src.config import STT_ENGINE, LANGUAGE, MIC_DEVICE, MIC_SR, MIC_CH
from src.engines import stt_router

app = FastAPI(title="EduNav+ STT Command Listener", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"status": "ok", "engine": STT_ENGINE}

@app.post("/listen")
def listen(
    duration: int = Query(6, ge=1, le=20),
    language: str = Query(LANGUAGE),
    device: int | None = Query(MIC_DEVICE),
    sr: int | None = Query(MIC_SR if MIC_SR else None),
    ch: int | None = Query(MIC_CH if MIC_CH else None),
):
    if STT_ENGINE != "google":
        raise HTTPException(status_code=400, detail="Mic capture is only enabled for engine=google. Use /stt for Vosk.")
    t0 = time.perf_counter()
    try:
        text, engine = stt_router.transcribe_mic(duration, language, sr, ch, device)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"command": text, "engine": engine, "latency_ms": latency_ms}
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {e}")

@app.post("/stt")
async def stt(
    file: UploadFile = File(...),
    language: str = Query(LANGUAGE)
):
    t0 = time.perf_counter()
    try:
        data = await file.read()
        text, engine = stt_router.transcribe_file(data, language)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"command": text, "engine": engine, "latency_ms": latency_ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {e}")
