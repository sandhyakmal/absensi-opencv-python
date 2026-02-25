from __future__ import annotations

import os, json
from typing import Dict, List, Optional

import numpy as np
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

DATA_DIR = os.getenv("FACE_API_DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "embeddings.npz")
META_PATH = os.path.join(DATA_DIR, "meta.json")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Face Attendance API", version="2.0.0")


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
def read_image_from_upload(file: UploadFile) -> np.ndarray:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
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
    np.savez_compressed(DB_PATH, **db)


def load_meta() -> dict:
    if not os.path.exists(META_PATH):
        return {"by_id": {}, "by_name": {}}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(meta: dict) -> None:
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def match_best(db: Dict[str, np.ndarray], emb_normed: np.ndarray) -> tuple[Optional[str], float]:
    """Return (best_id, best_similarity_cosine)."""
    if not db:
        return None, -1.0

    ids: List[str] = []
    mats: List[np.ndarray] = []
    for _id, e in db.items():
        ids.extend([_id] * e.shape[0])
        mats.append(e)

    mat = np.concatenate(mats, axis=0)  # (M,512)
    sims = mat @ emb_normed.astype(np.float32)  # cosine similarity, karena normed
    idx = int(np.argmax(sims))
    return ids[idx], float(sims[idx])


def sim_to_percent(sim: float) -> float:
    sim = max(0.0, min(1.0, sim))
    return float(round(sim * 100.0, 2))


# ---------- Schemas ----------
class EnrollResponse(BaseModel):
    ok: bool
    id: str
    name: str
    added: int           # embedding yang ditambahkan di request ini
    total: int           # total embedding untuk id ini setelah ditambah


class MatchResult(BaseModel):
    id: str
    name: str
    percent: float       # 0..100


class RecognizeResponse(BaseModel):
    ok: bool
    results: List[MatchResult] = []
    message: Optional[str] = None


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/enroll", response_model=EnrollResponse)
def enroll(
    id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    min_det_score: float = Form(0.5),
    reject_if_multiple_faces: bool = Form(True),
):
    """
    Enroll untuk absensi:
    - id unik
    - name tidak boleh sama (409 jika dipakai id lain)
    - panggil berkali-kali untuk id sama => total embedding bertambah
    """
    id = id.strip()
    name = name.strip()
    if not id:
        raise HTTPException(status_code=400, detail="id is required.")
    if not name:
        raise HTTPException(status_code=400, detail="name is required.")

    meta = load_meta()

    # Cegah name sama dipakai id lain
    existing_id_for_name = meta["by_name"].get(name)
    if existing_id_for_name and existing_id_for_name != id:
        raise HTTPException(status_code=409, detail=f"name already used by id={existing_id_for_name}")

    img = read_image_from_upload(file)
    faces = [f for f in face_app.get(img) if float(f.det_score) >= float(min_det_score)]
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected (or det_score too low).")

    if reject_if_multiple_faces and len(faces) != 1:
        raise HTTPException(status_code=400, detail=f"Expected exactly 1 face, found {len(faces)}")

    # ambil wajah terbesar (kalau reject_if_multiple_faces=False)
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True
    )
    emb = faces[0].normed_embedding  # (512,)

    db = load_db()
    embs = np.expand_dims(emb, axis=0)  # (1,512)

    if id in db:
        db[id] = np.concatenate([db[id], embs], axis=0)
    else:
        db[id] = embs

    save_db(db)

    # update meta
    meta["by_id"][id] = {"name": name}
    meta["by_name"][name] = id
    save_meta(meta)

    total = int(db[id].shape[0])
    return EnrollResponse(ok=True, id=id, name=name, added=1, total=total)


@app.post("/recognize", response_model=RecognizeResponse)
def recognize(
    file: UploadFile = File(...),
    threshold: float = Form(0.35),
    min_det_score: float = Form(0.5),
):
    """
    Return JSON hasil matching saja.
    Kalau tidak ada match, results = [] dan ok=false + message=NO_MATCH.
    """
    if not (0.0 < threshold < 1.0):
        raise HTTPException(status_code=400, detail="threshold must be between (0,1).")

    db = load_db()
    if not db:
        return RecognizeResponse(ok=False, results=[], message="DB_EMPTY")

    meta = load_meta()

    img = read_image_from_upload(file)
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
