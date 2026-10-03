"""Face detection with MTCNN."""
from dataclasses import dataclass

import cv2
import numpy as np
from mtcnn import MTCNN


@dataclass
class Face:
    box: tuple  # x, y, width, height
    confidence: float
    keypoints: dict


def load_detector() -> MTCNN:
    return MTCNN()


def detect_raw(detector: MTCNN, image: np.ndarray) -> list[Face]:
    """Every candidate MTCNN returns, best first, before any filtering."""
    raw = detector.detect_faces(
        image,
        min_face_size=20,
        threshold_pnet=0.5,
        threshold_rnet=0.7,
        threshold_onet=0.7,
    )
    faces = [Face(tuple(d["box"]), d["confidence"], d["keypoints"]) for d in raw]
    return sorted(faces, key=lambda f: f.confidence, reverse=True)


def classify(faces: list[Face], min_conf: float = 0.970, strict: bool = True) -> list[tuple[Face, str]]:
    """Label each candidate 'kept' or with the reason it was dropped."""
    kept: list[Face] = []
    out = []
    for f in faces:
        if f.confidence < min_conf:
            status = "dropped: low confidence"
        elif strict and not _plausible(f):
            status = "dropped: failed face-shape check"
        elif strict and any(_overlap(f.box, k.box) >= 0.3 for k in kept):
            status = "dropped: overlaps another face"
        else:
            status = "kept"
            kept.append(f)
        out.append((f, status))
    return out

def _plausible(face: Face) -> bool:
       """Eyes above nose, nose above mouth. Loose, so turned heads pass."""
       k = face.keypoints
       eye_y = (k["left_eye"][1] + k["right_eye"][1]) / 2
       mouth_y = (k["mouth_left"][1] + k["mouth_right"][1]) / 2
       return eye_y < k["nose"][1] < mouth_y


def _overlap(a: tuple, b: tuple) -> float:
    """Intersection area divided by the smaller box's area."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, min(a[2] * a[3], b[2] * b[3]))


def padded_box(image: np.ndarray, box: tuple, padding: int = 0) -> tuple:
    x, y, w, h = box
    h_img, w_img = image.shape[:2]
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(w_img, x + w + padding),
        min(h_img, y + h + padding),
    )


def crop(image: np.ndarray, face: Face, padding: int = 0) -> np.ndarray:
    x1, y1, x2, y2 = padded_box(image, face.box, padding)
    return image[y1:y2, x1:x2]


def annotate(image: np.ndarray, faces: list[Face], padding: int = 20, landmarks: bool = True) -> np.ndarray:
    out = image.copy()
    amber, blue = (255, 181, 71), (90, 170, 255)
    for i, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = padded_box(out, face.box, padding)
        cv2.rectangle(out, (x1, y1), (x2, y2), amber, 3)
        cv2.putText(out, f"{i}  {face.confidence:.2f}", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, amber, 2)
        if landmarks:
            for pt in face.keypoints.values():
                cv2.circle(out, tuple(int(v) for v in pt), 5, blue, -1)
    return out
