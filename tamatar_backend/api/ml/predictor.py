import pickle
import numpy as np
from .feature_extractor import extract_features
from pathlib import Path
import json
import threading

_model_obj = None
classifier = None
scaler = None
selected_features_idx = None
label_classes = None
idx_to_class = None
_load_lock = threading.Lock()

def _probs_from_classifier(clf, features_2d):
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(features_2d)[0]
        return probs.astype(float)
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(features_2d)
        scores = np.atleast_2d(scores)
        if scores.shape[1] == 1:
            s = scores[:, 0]
            scores = np.vstack([-s, s]).T
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs[0].astype(float)
    return None

def _load_model():
    global _model_obj, classifier, scaler, selected_features_idx, label_classes, idx_to_class
    if _model_obj is not None:
        return

    with _load_lock:
        if _model_obj is not None:
            return

        models_dir = Path(__file__).resolve().parent / "models"
        model_path = models_dir / "LSVM_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with model_path.open("rb") as f:
            _model_obj = pickle.load(f)

        # validate expected keys
        for key in ("model", "scaler", "selected_features", "label_classes"):
            if key not in _model_obj:
                raise KeyError(f"Missing '{key}' in model object; keys: {list(_model_obj.keys())}")

        classifier = _model_obj["model"]
        scaler = _model_obj["scaler"]
        selected_features_idx = np.asarray(_model_obj["selected_features"], dtype=int)
        label_classes = _model_obj["label_classes"]

        class_map_path = models_dir / "class_mapping.json"
        if not class_map_path.exists():
            raise FileNotFoundError(f"Class mapping file not found: {class_map_path}")
        with class_map_path.open() as f:
            class_to_idx = json.load(f)

        # normalize mapping keys to ints if possible
        try:
            idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        except Exception:
            # fallback: keep original mapping (may be string keys)
            idx_to_class = {v: k for k, v in class_to_idx.items()}

def predict_image(image_path):
    try:
        _load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    features = extract_features(image_path)
    features = np.asarray(features, dtype=float)

    if selected_features_idx is None:
        raise RuntimeError("selected_features not loaded from model")

    if features.ndim != 1:
        raise ValueError("Extracted features must be a 1-D vector")

    if np.any(selected_features_idx >= features.size) or np.any(selected_features_idx < 0):
        raise IndexError("selected_features indices out of range for extracted features")

    selected = features[selected_features_idx]
    scaled = scaler.transform([selected])

    try:
        pred = classifier.predict(scaled)[0]
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    probs = None
    try:
        probs = _probs_from_classifier(classifier, scaled)
    except Exception:
        probs = None

    confidence = float(np.max(probs)) if isinstance(probs, np.ndarray) else None

    # safe label mapping: try int pred first, else str
    label = None
    try:
        label = idx_to_class.get(int(pred))
    except Exception:
        label = None
    if label is None:
        label = idx_to_class.get(str(pred)) if isinstance(idx_to_class, dict) else None
    if label is None:
        label = str(pred)

    return {
        "prediction": label,
        "confidence": confidence,
        "class_index": int(pred)
    }