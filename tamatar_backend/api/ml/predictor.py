import pickle
import numpy as np
from .feature_extractor import extract_features
from pathlib import Path
import json

_model_obj = None
classifier = None
scaler = None
selected_features_idx = None
label_classes = None
idx_to_class = None 

def _load_model():
    global _model_obj, classifier, scaler, selected_features_idx, label_classes, idx_to_class
    if _model_obj is not None:
        return

    models_dir = Path(__file__).resolve().parent / "models"
    model_path = models_dir / "LSVM_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with model_path.open("rb") as f:
        _model_obj = pickle.load(f)

    classifier = _model_obj["model"]
    scaler = _model_obj["scaler"]
    selected_features_idx = _model_obj["selected_features"]
    label_classes = _model_obj["label_classes"]

    with open(models_dir / "class_mapping.json") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}


def predict_image(image_path):
    _load_model()

    features = extract_features(image_path)

    features = np.array(features)[selected_features_idx]

    features = scaler.transform([features])

    pred = classifier.predict(features)[0]
    
    probs = classifier.predict_proba(features)[0]
    confidence = float(np.max(probs))
    
    label = idx_to_class[pred]

    return {
        "prediction": label,
        "confidence": confidence,
        "class_index": int(pred)
    }
