from __future__ import annotations

"""
auth/train_face.py
------------------
Interactive face-capture and LBPH training utility.

Usage (run directly)::

    python auth/train_face.py                        # prompts for user label
    python auth/train_face.py --label owner          # skip the label prompt
    python auth/train_face.py --label owner --samples 100 --camera 1

Workflow
--------
1. Opens the webcam and shows a live preview window.
2. Detects the face with a Haar cascade on every frame.
3. Collects ``--samples`` (default 50) cropped grayscale ROI images,
   saving them to ``auth/known_faces/<label>/``.
4. Trains an LBPH recognizer on **all** images currently in ``known_faces/``
   (so calling this script multiple times for different users accumulates data).
5. Saves the model to ``models/face_model.yml``.
6. Displays a live progress bar overlay in the preview window.

Press  Q  at any time to abort.
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Optional OpenCV import
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
except ImportError:
    print("[ERROR] OpenCV is not installed.  Run:  pip install opencv-python")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the script works from any cwd)
# ---------------------------------------------------------------------------
_AUTH_DIR   = Path(__file__).resolve().parent                  # jarvis-assistant/auth/
_ROOT_DIR   = _AUTH_DIR.parent                                 # jarvis-assistant/
_MODEL_DIR  = _ROOT_DIR / "models"
_MODEL_PATH = _MODEL_DIR / "face_model.yml"
_FACES_DIR  = _AUTH_DIR / "known_faces"

# ---------------------------------------------------------------------------
# Haar cascade path (bundled with opencv-python)
# ---------------------------------------------------------------------------

def _cascade_xml() -> str:
    base = getattr(cv2, "data", None)
    if base and hasattr(base, "haarcascades"):
        p = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if p.exists():
            return str(p)
    # Fallback: search the cv2 package tree
    for candidate in Path(cv2.__file__).parent.rglob("haarcascade_frontalface_default.xml"):
        return str(candidate)
    raise FileNotFoundError(
        "haarcascade_frontalface_default.xml not found.  "
        "Reinstall opencv-python or opencv-python-headless."
    )


# ---------------------------------------------------------------------------
# Visual progress-bar drawn onto the OpenCV frame
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame: "np.ndarray",
    label: str,
    captured: int,
    total: int,
    faces: list,
) -> "np.ndarray":
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # -- Draw detected face rectangles --
    for (x, y, fw, fh) in faces:
        cv2.rectangle(overlay, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

    # -- Progress bar background (bottom strip) --
    bar_h      = 50
    bar_y      = h - bar_h
    cv2.rectangle(overlay, (0, bar_y), (w, h), (30, 30, 30), -1)

    # -- Filled progress bar --
    filled_w = int(w * (captured / total))
    cv2.rectangle(overlay, (0, bar_y + 8), (filled_w, h - 8), (0, 210, 0), -1)

    # -- Text --
    pct  = int(100 * captured / total)
    info = f"User: {label}   Captured: {captured}/{total}  ({pct}%)"
    cv2.putText(
        overlay, info,
        (10, bar_y + 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )

    # -- Status hint at top --
    if len(faces) == 0:
        hint, colour = "No face detected – adjust position", (0, 100, 255)
    elif captured >= total:
        hint, colour = "Capture complete!  Training …", (0, 255, 200)
    else:
        hint, colour = "Hold still …", (0, 255, 0)

    cv2.putText(
        overlay, hint,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
    )

    return overlay


# ---------------------------------------------------------------------------
# Step 1 – collect face samples
# ---------------------------------------------------------------------------

def capture_samples(
    label: str,
    n_samples: int = 50,
    camera_index: int = 0,
    min_face_size: tuple[int, int] = (80, 80),
) -> list[Path]:
    """
    Open the webcam and save *n_samples* face crops under
    ``auth/known_faces/<label>/``.

    Returns the list of saved image paths.
    """
    save_dir = _FACES_DIR / label
    save_dir.mkdir(parents=True, exist_ok=True)

    cascade = cv2.CascadeClassifier(_cascade_xml())
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}.")

    saved_paths: list[Path] = []
    captured = 0
    # Small cooldown between saves to increase sample diversity
    _last_save = 0.0
    _SAVE_INTERVAL = 0.12   # seconds

    window_name = "JARVIS – Face Capture  (press Q to abort)"
    logger.info(f"Starting face capture: label='{label}', target={n_samples} samples.")

    try:
        while captured < n_samples:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Dropped frame.")
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=min_face_size,
            )

            now = time.monotonic()
            if len(faces) > 0 and (now - _last_save) >= _SAVE_INTERVAL:
                # Use the largest detected face
                (x, y, fw, fh) = max(faces, key=lambda r: r[2] * r[3])
                roi = gray[y : y + fh, x : x + fw]
                roi_resized = cv2.resize(roi, (200, 200))

                img_path = save_dir / f"{label}_{captured:04d}.jpg"
                cv2.imwrite(str(img_path), roi_resized)
                saved_paths.append(img_path)
                captured += 1
                _last_save = now

                logger.debug(f"Saved sample {captured}/{n_samples} → {img_path.name}")

            display = _draw_overlay(frame, label, captured, n_samples, faces)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                logger.warning("Capture aborted by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info(f"Capture done: {len(saved_paths)} samples saved to {save_dir}")
    return saved_paths


# ---------------------------------------------------------------------------
# Step 2 – train LBPH model on all known-faces data
# ---------------------------------------------------------------------------

def train_model() -> bool:
    """
    Train an LBPH recognizer from **all** images in ``auth/known_faces/``
    and save the model to ``models/face_model.yml``.

    Returns True on success.
    """
    cascade = cv2.CascadeClassifier(_cascade_xml())

    faces:  list[np.ndarray] = []
    labels: list[int]        = []
    label_map: dict[str, int] = {}

    if not _FACES_DIR.exists() or not any(_FACES_DIR.iterdir()):
        logger.error(f"No training data found under {_FACES_DIR}.")
        return False

    for idx, user_dir in enumerate(sorted(_FACES_DIR.iterdir())):
        if not user_dir.is_dir():
            continue
        label_map[user_dir.name] = idx
        img_count = 0

        for img_path in user_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Images saved by this script are already cropped face ROIs;
            # run detection anyway to handle hand-placed training images.
            detected = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
            )
            if len(detected) > 0:
                (x, y, fw, fh) = max(detected, key=lambda r: r[2] * r[3])
                roi = cv2.resize(gray[y : y + fh, x : x + fw], (200, 200))
            else:
                # Already-cropped ROI (no face detected → use whole image)
                roi = cv2.resize(gray, (200, 200))

            faces.append(roi)
            labels.append(idx)
            img_count += 1

        logger.info(f"  Loaded {img_count} images for '{user_dir.name}' (label {idx})")

    if not faces:
        logger.error("No usable face samples were found.")
        return False

    print(f"\nTraining LBPH recognizer on {len(faces)} samples …", flush=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(_MODEL_PATH))

    logger.success(f"Model saved → {_MODEL_PATH}")
    print(f"\n✓ Model saved to  {_MODEL_PATH}")
    print(f"  Users trained: {list(label_map.keys())}\n")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture face samples and train the LBPH recognizer for JARVIS."
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="User label (sub-folder name under auth/known_faces/).  "
             "Prompted interactively if omitted.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of face samples to capture (default: 50).",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Skip capture; retrain the model from existing images.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.train_only:
        label = args.label
        if not label:
            label = input("Enter a label for this user (e.g. 'owner'): ").strip()
        if not label:
            print("[ERROR] No label provided.  Exiting.")
            sys.exit(1)

        print(f"\n[JARVIS Face Capture]")
        print(f"  Label   : {label}")
        print(f"  Samples : {args.samples}")
        print(f"  Camera  : {args.camera}")
        print(f"  Save to : {_FACES_DIR / label}\n")
        print("  Look at the camera.  Press Q to abort.\n")

        try:
            saved = capture_samples(
                label=label,
                n_samples=args.samples,
                camera_index=args.camera,
            )
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        if len(saved) < args.samples:
            print(f"\n[WARNING] Only {len(saved)}/{args.samples} samples captured.")
            if not saved:
                print("No samples saved – training aborted.")
                sys.exit(1)

    success = train_model()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
