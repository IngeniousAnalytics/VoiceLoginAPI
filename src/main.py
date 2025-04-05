#---------------------------
# main.py
#---------------------------
import sys
import os
import logging
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import tempfile, subprocess
from pathlib import Path
import numpy as np
from database import SessionLocal, engine
from models import Base, UserEmbedding
from sqlalchemy import select
from fastapi.responses import JSONResponse

# Setup logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "verification.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
MIN_RECORDING_DURATION = 3  # seconds
SIMILARITY_THRESHOLD = 0.8

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_ffmpeg_path():
    if sys.platform == "win32":
        for path in [
            r"C:\\ffmpeg\\bin\\ffmpeg.exe",
            r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            r"C:\\Tools\\ffmpeg\\bin\\ffmpeg.exe"
        ]:
            if os.path.exists(path):
                return path
    return "ffmpeg"

def convert_audio(input_path: str, output_path: str):
    ffmpeg_path = get_ffmpeg_path()
    subprocess.run([
        ffmpeg_path, "-i", input_path,
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        "-y", output_path
    ], check=True)

@app.post("/register")
async def register_user(username: str = Form(...), audio: UploadFile = File(...)):
    if len(username) < 3:
        raise HTTPException(400, detail="Username must be ≥3 characters")

    async with SessionLocal() as session:
        result = await session.execute(select(UserEmbedding).where(UserEmbedding.username == username))
        if result.scalar():
            raise HTTPException(400, detail="Username already exists")

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            input_path = tmp.name

        wav_path = Path(input_path).with_suffix(".wav")
        convert_audio(input_path, str(wav_path))

        waveform, sample_rate = torchaudio.load(wav_path)
        duration = waveform.shape[1] / sample_rate

        if duration < MIN_RECORDING_DURATION:
            raise HTTPException(400, detail="Recording too short")

        embedding = classifier.encode_batch(waveform).squeeze().numpy().tolist()

        user = UserEmbedding(username=username, embedding=embedding)
        session.add(user)
        await session.commit()

        for path in [input_path, wav_path]:
            try:
                os.remove(path)
            except:
                pass

        return {"status": "success", "username": username, "duration": round(duration, 2)}

@app.post("/verify")
async def verify_user(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        input_path = tmp.name

    wav_path = Path(input_path).with_suffix(".wav")
    convert_audio(input_path, str(wav_path))
    waveform, sample_rate = torchaudio.load(wav_path)

    if waveform.shape[1] / sample_rate < 1.0:
        raise HTTPException(400, detail="Audio too short")

    input_embedding = classifier.encode_batch(waveform).squeeze().numpy()

    async with SessionLocal() as session:
        result = await session.execute(select(UserEmbedding))
        users = result.scalars().all()

        best_match = None
        highest_similarity = 0.0

        for user in users:
            similarity = cosine_similarity(input_embedding, np.array(user.embedding))
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = user.username

        for path in [input_path, wav_path]:
            try:
                os.remove(path)
            except:
                pass

        log_message = f"Attempted match: {best_match if best_match else 'No match'} | Similarity: {round(float(highest_similarity), 4)}"
        logging.info(log_message)

        if highest_similarity >= SIMILARITY_THRESHOLD:
            return JSONResponse(content={"authenticated": True, "username": best_match, "similarity": round(float(highest_similarity), 4)})

        return JSONResponse(content={"authenticated": False, "similarity": round(float(highest_similarity), 4)})

@app.get("/users")
async def list_users():
    async with SessionLocal() as session:
        result = await session.execute(select(UserEmbedding.username))
        return {"users": [r[0] for r in result.all()]}

@app.get("/health")
async def health_check():
    try:
        async with SessionLocal() as session:
            await session.execute(select(1))
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/test-db")
async def test_db():
    try:
        async with SessionLocal() as session:
            await session.execute(select(1))
        return {"status": "Database connection successful"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
