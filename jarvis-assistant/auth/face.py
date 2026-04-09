from __future__ import annotations

"""
auth/face.py
Face authentication using OpenCV LBPH recognizer + Haar cascade detector.

Workflow
--------
1. On first call (or when the model is stale) the module trains an LBPH
   recognizer from grayscale face images stored under
   ``config.KNOWN_FACES_DIR/<user_label>/*.jpg|png``.
2. ``verify()`` opens the default camera, captures a frame, detects faces
   with the frontal-face Haar cascade, and passes each detected face to the
   trained recognizer.
3. Returns True only when at least one detected face yields a confidence
   score ≤ ``threshold`` (lower = better match in LBPH parlance).

Directory layout expected::

    auth/
        known_faces/
            owner/          ← any sub-folder name is the user label
                img1.jpg
                img2.png
                ...
        face_model.yml      ← auto-generated; re-trained when missing

Configuration (from config.py / .env)
--------------------------------------
* ``KNOWN_FACES_DIR``           – path to labelled training images
* ``FACE_RECOGNITION_THRESHOLD``– confidence threshold (LBPH score ≤ → auth)
                                  Default: 80.  Range: lower is stricter.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Optional OpenCV import (graceful degradation when cv2 is not installed)
# ---------------------------------------------------------------------------
try:
    import cv2  # type: ignore
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) is not installed – face auth is unavailable.")

from config import config

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_MODEL_PATH: Path = Path(config.KNOWN_FACES_DIR).parent / "face_model.yml"

# LBPH outputs a *distance* score; lower means better match.
# config.FACE_RECOGNITION_THRESHOLD is a float in [0,1] (kept for API compat)
# so we map it to the 0–300 LBPH distance space:  threshold_lbph = (1 - t) * 150
# With the default 0.6 in config that gives 60.0, which is fairly strict.
def _lbph_threshold() -> float:
    t = float(getattr(config, "FACE_RECOGNITION_THRESHOLD", 0.6))
    # Allow override with a raw LBPH value > 1 in the env
    if t > 1.0:
        return t
    return (1.0 - t) * 150.0


# ---------------------------------------------------------------------------
# Haar cascade
# ---------------------------------------------------------------------------

def _cascade_path() -> str:
    """Return the absolute path to haarcascade_frontalface_default.xml."""
    if not _CV2_AVAILABLE:
        return ""
    # cv2.data.haarcascades is available in opencv-python ≥ 4.x
    base = getattr(cv2, "data", None)
    if base and hasattr(base, "haarcascades"):
        return os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    # Fallback: search relative to cv2 package
    cv2_dir = Path(cv2.__file__).parent
    candidates = list(cv2_dir.rglob("haarcascade_frontalface_default.xml"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(
        "haarcascade_frontalface_default.xml not found. "
        "Re-install opencv-python or opencv-python-headless."
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _load_training_data() -> tuple[list, list]:
    """
    Walk ``KNOWN_FACES_DIR`` and return (faces, labels) suitable for LBPH training.
    Each sub-folder is treated as a distinct user label (encoded as an int index).
    """
    faces: list = []
    labels: list = []
    label_map: dict[str, int] = {}
    known_dir = Path(config.KNOWN_FACES_DIR)

    if not known_dir.exists() or not any(known_dir.iterdir()):
        logger.warning(f"No training images found in {known_dir}")
        return faces, labels

    cascade = cv2.CascadeClassifier(_cascade_path())

    for idx, user_dir in enumerate(sorted(known_dir.iterdir())):
        if not user_dir.is_dir():
            continue
        label_map[user_dir.name] = idx
        img_paths = [
            p for p in user_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not img_paths:
            logger.debug(f"No images in {user_dir}")
            continue

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Cannot read image: {img_path}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            for (x, y, w, h) in detected:
                faces.append(gray[y : y + h, x : x + w])
                labels.append(idx)

    logger.info(
        f"Training data loaded: {len(faces)} faces across {len(label_map)} user(s) "
        f"{list(label_map.keys())}"
    )
    return faces, labels


def train_model(force: bool = False) -> bool:
    """
    Train and persist the LBPH recognizer.

    Parameters
    ----------
    force : bool
        Re-train even if ``face_model.yml`` already exists.

    Returns True on success.
    """
    if not _CV2_AVAILABLE:
        logger.error("cv2 not available – cannot train model.")
        return False

    if _MODEL_PATH.exists() and not force:
        logger.debug("Face model already exists; skipping training.")
        return True

    faces, labels = _load_training_data()
    if not faces:
        logger.error("No face samples found – cannot train the recognizer.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(_MODEL_PATH))
    logger.success(f"Face model trained and saved to {_MODEL_PATH}")
    return True


# ---------------------------------------------------------------------------
# Recognizer loader (cached)
# ---------------------------------------------------------------------------

_recognizer: Optional["cv2.face.LBPHFaceRecognizer"] = None  # type: ignore


def _get_recognizer() -> Optional["cv2.face.LBPHFaceRecognizer"]:  # type: ignore
    global _recognizer
    if _recognizer is not None:
        return _recognizer
    if not _MODEL_PATH.exists():
        logger.info("Model not found – triggering training.")
        if not train_model():
            return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(_MODEL_PATH))
    _recognizer = recognizer
    logger.info("LBPH face recognizer loaded from disk.")
    return _recognizer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify(
    threshold: Optional[float] = None,
    camera_index: int = 0,
    max_attempts: int = 30,
) -> bool:
    """
    Capture a live frame from the webcam and verify the face against the
    trained model.

    Parameters
    ----------
    threshold : float | None
        LBPH confidence upper bound.  Scores *below* this value are accepted.
        Defaults to the value derived from ``config.FACE_RECOGNITION_THRESHOLD``.
    camera_index : int
        OpenCV camera device index (default 0 = built-in webcam).
    max_attempts : int
        Number of consecutive frames to try before giving up (avoids
        hanging if no face is detected immediately).

    Returns
    -------
    bool
        True  – a recognized face was found within the threshold.
        False – no match (or cv2 unavailable / model missing).
    """
    if not _CV2_AVAILABLE:
        logger.error("OpenCV is not installed – face verification unavailable.")
        return False

    if not config.FACE_AUTH_ENABLED:
        logger.info("Face auth is disabled in config – skipping verification.")
        return True   # gracefully bypass when feature is off

    limit = threshold if threshold is not None else _lbph_threshold()
    recognizer = _get_recognizer()
    if recognizer is None:
        logger.error("Recognizer could not be loaded – denying access.")
        return False

    cascade = cv2.CascadeClassifier(_cascade_path())

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera at index {camera_index}.")
        return False

    authorized = False
    attempts = 0

    try:
        logger.info("Face verification started – please look at the camera.")
        while attempts < max_attempts:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read camera frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            for (x, y, w, h) in faces:
                roi = gray[y : y + h, x : x + w]
                label, confidence = recognizer.predict(roi)
                logger.debug(f"Prediction → label={label}, confidence={confidence:.2f}")
                if confidence <= limit:
                    logger.success(
                        f"Face authorized (confidence={confidence:.2f} ≤ threshold={limit:.2f})"
                    )
                    authorized = True
                    break
                else:
                    logger.debug(
                        f"Face rejected (confidence={confidence:.2f} > threshold={limit:.2f})"
                    )

            if authorized:
                break
            attempts += 1

    finally:
        cap.release()

    if not authorized:
        logger.warning("Face verification failed – access denied.")

    return authorized


def retrain() -> bool:
    """Force a full retrain of the face model (e.g. after adding new images)."""
    global _recognizer
    _recognizer = None          # invalidate cache
    return train_model(force=True)
