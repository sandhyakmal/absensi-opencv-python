from __future__ import annotations

import os
import json
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


DATA_DIR = os.getenv("FACE_API_DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "embeddings.npz")
META_PATH = os.path.join(DATA_DIR, "meta.json")


def init_face_app() -> FaceAnalysis:
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else []
    providers.append("CPUExecutionProvider")

    fa = FaceAnalysis(name="buffalo_l", providers=providers)
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    fa.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return fa


def load_db() -> Dict[str, np.ndarray]:
    if not os.path.exists(DB_PATH):
        return {}
    data = np.load(DB_PATH, allow_pickle=True)
    return {k: data[k] for k in data.files}  # id -> (N,512)


def load_meta() -> dict:
    if not os.path.exists(META_PATH):
        return {"by_id": {}, "by_name": {}}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def match_best(db: Dict[str, np.ndarray], emb_normed: np.ndarray) -> Tuple[Optional[str], float]:
    if not db:
        return None, -1.0

    best_id = None
    best_sim = -1.0
    v = emb_normed.astype(np.float32)

    for _id, embs in db.items():        # embs: (N,512)
        sims = embs @ v                 # cosine similarity karena normed_embedding
        m = float(np.max(sims))
        if m > best_sim:
            best_sim = m
            best_id = _id

    return best_id, best_sim


def sim_to_percent(sim: float) -> float:
    sim = max(0.0, min(1.0, sim))
    return float(round(sim * 100.0, 2))


def main():
    threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.35"))
    min_det_score = float(os.getenv("MIN_DET_SCORE", "0.5"))
    cam_index = int(os.getenv("CAM_INDEX", "0"))

    face_app = init_face_app()

    db = load_db()
    meta = load_meta()

    if not db:
        print("DB_EMPTY: embeddings.npz belum ada. Enroll dulu lewat API /enroll.")
        return

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Gagal buka kamera index={cam_index}")
        return

    print("Kamera jalan. Tekan 'q' untuk keluar.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Deteksi wajah
        faces = [f for f in face_app.get(frame) if float(f.det_score) >= min_det_score]

        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            best_id, sim = match_best(db, f.normed_embedding)

            label = "UNKNOWN"
            percent = 0.0

            if best_id is not None and sim >= threshold:
                name = meta.get("by_id", {}).get(best_id, {}).get("name", best_id)
                percent = sim_to_percent(sim)
                label = f"{name} ({percent}%)"

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Face Recognition - Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()