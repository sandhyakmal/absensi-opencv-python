from __future__ import annotations

import os, json
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis

import base64
import hashlib

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends
from pydantic import BaseModel
from filelock import FileLock
from dotenv import load_dotenv

import mysql.connector
from mysql.connector import pooling

# Load .env (DEV friendly)
load_dotenv()

# =========================
# Storage config
# =========================
DATA_DIR = os.getenv("FACE_API_DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "embeddings.npz")
META_PATH = os.path.join(DATA_DIR, "meta.json")
LOCK_PATH = os.path.join(DATA_DIR, "db.lock")

MAX_EMB_PER_ID = int(os.getenv("MAX_EMB_PER_ID", "20"))

os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Face Attendance API", version="2.1.0")

# =========================
# MySQL config
# =========================
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "8889"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DB = os.getenv("MYSQL_DB", "db_opencv")
MYSQL_POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", "5"))
MYSQL_CONNECT_TIMEOUT = int(os.getenv("MYSQL_CONNECT_TIMEOUT", "5"))

# Optional fallback (kalau kamu mau)
FALLBACK_API_KEY = os.getenv("FACE_API_KEY")  # optional; boleh kosong

db_pool = pooling.MySQLConnectionPool(
    pool_name="face_api_pool",
    pool_size=MYSQL_POOL_SIZE,
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DB,
    autocommit=True,
    connection_timeout=MYSQL_CONNECT_TIMEOUT,
)

# ---------- Security (API Key) ----------
def _decode_base64_key(x_api_key: str) -> str:
    try:
        # strip whitespace, validate base64
        s = (x_api_key or "").strip()
        plain = base64.b64decode(s, validate=True).decode("utf-8")
        return plain
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid API key encoding (expected base64).")

def is_api_key_valid_db(plain_key: str) -> bool:
    key_hash = hashlib.sha256(plain_key.encode("utf-8")).hexdigest()

    cnx = db_pool.get_connection()
    cur = None
    try:
        cur = cnx.cursor()
        cur.execute(
            "SELECT 1 FROM api_keys WHERE active=1 AND key_hash=%s LIMIT 1",
            (key_hash,),
        )
        ok = cur.fetchone() is not None

        if ok:
            cur.execute(
                "UPDATE api_keys SET last_used_at=NOW() WHERE key_hash=%s",
                (key_hash,),
            )
        return ok
    finally:
        try:
            if cur:
                cur.close()
        except Exception:
            pass
        cnx.close()

def require_api_key(x_api_key: str = Header(default=None)):
    """
    Client kirim X-API-KEY dalam BASE64.
    Server decode base64 -> plain_key -> SHA256 -> cek ke MySQL.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized - Header tidak ditemukan: X-API-KEY")

    plain_key = _decode_base64_key(x_api_key)

    # 1) Cek ke DB
    try:
        if is_api_key_valid_db(plain_key):
            return
        raise HTTPException(status_code=401, detail="Unauthorized")
    except mysql.connector.Error:
        # 2) Kalau DB error, optional fallback (kalau diset)
        if FALLBACK_API_KEY and plain_key == FALLBACK_API_KEY:
            return
        raise HTTPException(status_code=503, detail="Auth DB unavailable")

# ---------- Model init ----------
def init_face_app() -> FaceAnalysis:
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else []
    providers.append("CPUExecutionProvider")

    fa = FaceAnalysis(name="buffalo_l", providers=providers)
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    fa.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return fa

face_app = init_face_app()

# ---------- Helpers ----------
async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    if file.content_type not in (None, "", "image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Invalid content_type. Use image/jpeg or image/png.")

    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image. Use JPG/PNG.")
    return img

def load_db() -> Dict[str, np.ndarray]:
    if not os.path.exists(DB_PATH):
        return {}
    data = np.load(DB_PATH, allow_pickle=True)
    return {k: data[k] for k in data.files}

def save_db(db: Dict[str, np.ndarray]) -> None:
    tmp = DB_PATH + ".tmp.npz"          # penting: akhiri dengan .npz
    np.savez_compressed(tmp, **db)
    os.replace(tmp, DB_PATH)

def load_meta() -> dict:
    if not os.path.exists(META_PATH):
        return {"by_id": {}, "by_name": {}}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_meta(meta: dict) -> None:
    tmp = META_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, META_PATH)

def match_best(db: Dict[str, np.ndarray], emb_normed: np.ndarray) -> Tuple[Optional[str], float]:
    if not db:
        return None, -1.0

    best_id = None
    best_sim = -1.0
    v = emb_normed.astype(np.float32)

    for _id, embs in db.items():  # embs: (N,512)
        sims = embs @ v
        m = float(np.max(sims))
        if m > best_sim:
            best_sim = m
            best_id = _id

    return best_id, best_sim

def sim_to_percent(sim: float) -> float:
    sim = max(0.0, min(1.0, sim))
    return float(round(sim * 100.0, 2))

# ---------- Schemas ----------
class EnrollResponse(BaseModel):
    ok: bool
    id: str
    name: str
    added: int
    total: int

class MatchResult(BaseModel):
    id: str
    name: str
    percent: float

class RecognizeResponse(BaseModel):
    ok: bool
    results: List[MatchResult] = []
    message: Optional[str] = None

class DeleteResponse(BaseModel):
    ok: bool
    id: str
    removed: bool
    message: Optional[str] = None

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/enroll", response_model=EnrollResponse)
async def enroll(
    id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    min_det_score: float = Form(0.5),
    reject_if_multiple_faces: bool = Form(True),
    _: None = Depends(require_api_key),
):
    id = id.strip()
    name = name.strip()
    if not id:
        raise HTTPException(status_code=400, detail="id is required.")
    if not name:
        raise HTTPException(status_code=400, detail="name is required.")

    img = await read_image_from_upload(file)
    faces = [f for f in face_app.get(img) if float(f.det_score) >= float(min_det_score)]
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected (or det_score too low).")

    if reject_if_multiple_faces and len(faces) != 1:
        raise HTTPException(status_code=400, detail=f"Expected exactly 1 face, found {len(faces)}")

    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    emb = faces[0].normed_embedding  # (512,)

    with FileLock(LOCK_PATH):
        meta = load_meta()

        existing_id_for_name = meta["by_name"].get(name)
        if existing_id_for_name and existing_id_for_name != id:
            raise HTTPException(status_code=409, detail=f"name already used by id={existing_id_for_name}")

        db = load_db()
        embs = np.expand_dims(emb, axis=0)

        if id in db:
            db[id] = np.concatenate([db[id], embs], axis=0)
            if db[id].shape[0] > MAX_EMB_PER_ID:
                db[id] = db[id][-MAX_EMB_PER_ID:]
        else:
            db[id] = embs

        save_db(db)

        meta["by_id"][id] = {"name": name}
        meta["by_name"][name] = id
        save_meta(meta)

        total = int(db[id].shape[0])

    return EnrollResponse(ok=True, id=id, name=name, added=1, total=total)

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(...),
    threshold: float = Form(0.35),
    min_det_score: float = Form(0.5),
    _: None = Depends(require_api_key),
):
    if not (0.0 < threshold < 1.0):
        raise HTTPException(status_code=400, detail="threshold must be between (0,1).")

    with FileLock(LOCK_PATH):
        db = load_db()
        meta = load_meta()

    if not db:
        return RecognizeResponse(ok=False, results=[], message="DB_EMPTY")

    img = await read_image_from_upload(file)
    faces = [f for f in face_app.get(img) if float(f.det_score) >= float(min_det_score)]
    if not faces:
        return RecognizeResponse(ok=False, results=[], message="NO_FACE")

    results: List[MatchResult] = []
    for f in faces:
        best_id, sim = match_best(db, f.normed_embedding)
        if best_id is not None and sim >= threshold:
            name = meta["by_id"].get(best_id, {}).get("name", "")
            results.append(MatchResult(id=best_id, name=name, percent=sim_to_percent(sim)))

    if not results:
        return RecognizeResponse(ok=False, results=[], message="NO_MATCH")

    return RecognizeResponse(ok=True, results=results, message=None)

@app.post("/delete", response_model=DeleteResponse)
def delete_face(
    id: str = Form(...),
    _: None = Depends(require_api_key),
):
    id = id.strip()
    if not id:
        raise HTTPException(status_code=400, detail="id is required.")

    with FileLock(LOCK_PATH):
        db = load_db()
        meta = load_meta()

        # cek ada tidak
        existed_in_db = id in db
        existed_in_meta = id in meta.get("by_id", {})

        # kalau tidak ada sama sekali
        if not existed_in_db and not existed_in_meta:
            return DeleteResponse(ok=True, id=id, removed=False, message="NOT_FOUND")

        # hapus embedding
        if existed_in_db:
            del db[id]
            save_db(db)

        # hapus meta by_id + by_name yang menunjuk ke id ini
        if existed_in_meta:
            name = meta["by_id"].get(id, {}).get("name")
            meta["by_id"].pop(id, None)

            # by_name hanya dihapus kalau masih menunjuk ke id ini (biar aman)
            if name:
                if meta.get("by_name", {}).get(name) == id:
                    meta["by_name"].pop(name, None)

            save_meta(meta)

    return DeleteResponse(ok=True, id=id, removed=True, message="DELETED")