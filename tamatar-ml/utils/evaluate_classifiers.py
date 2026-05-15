import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.base import clone


def find_features_dir(root: Path) -> Path:
    cand1 = root / "features" / "tamatar"
    cand2 = root / "features"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"No features directory found. Checked: {cand1} and {cand2}")


def evaluate_on_dataset(features_dir: Path, classifier_files, results, do_cv: int | None = None):
    X_raw = np.load(features_dir / "combined_features.npy")
    y_raw = np.load(features_dir / "labels.npy")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Try loading class mapping (maps label id -> human name)
    class_map_path = features_dir / "class_mapping.json"
    class_map = None
    if class_map_path.exists():
        try:
            with open(class_map_path, "r") as fh:
                class_map = json.load(fh)
        except Exception:
            class_map = None
    # Normalize mapping so it maps numeric id -> human name
    if class_map is not None:
        # If mapping is name->id (as in the file), invert it
        # detect by checking a single key type
        try:
            sample_key = next(iter(class_map.keys()))
            sample_val = class_map[sample_key]
            if isinstance(sample_key, str) and isinstance(sample_val, int):
                # invert
                class_map = {int(v): k for k, v in class_map.items()}
        except StopIteration:
            class_map = None

    for clf_file in classifier_files:
        print("\n" + "=" * 70)
        print(f"Evaluating {clf_file}")
        print("=" * 70)

        with open(features_dir / clf_file, "rb") as f:
            save_object = pickle.load(f)

        classifier = save_object["model"]
        scaler = save_object.get("scaler")
        selected_features = save_object["selected_features"]
        # Build human-readable class names using class_map if available.
        raw_label_classes = save_object["label_classes"]
        class_names = []
        for c in raw_label_classes:
            name = None
            if class_map is not None:
                # try string key then int key
                name = class_map.get(str(c)) if isinstance(class_map, dict) else None
                if name is None:
                    try:
                        name = class_map.get(int(c))
                    except Exception:
                        name = None
            if name is None:
                name = str(c)
            class_names.append(name)

        X_selected = X_raw[:, selected_features]

        if do_cv:
            # Run stratified cross-validation using a fresh StandardScaler per fold
            print(f"Running Stratified {do_cv}-fold CV (no leakage from saved scaler)...")
            pipeline = make_pipeline(StandardScaler(), clone(classifier))
            cv = StratifiedKFold(n_splits=do_cv, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X_selected, y, cv=cv, scoring="accuracy", n_jobs=-1)
            y_pred = cross_val_predict(pipeline, X_selected, y, cv=cv, n_jobs=-1)
            print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
            accuracy = accuracy_score(y, y_pred)
        else:
            # Use saved scaler + final fitted model
            if scaler is not None:
                X_scaled = scaler.transform(X_selected)
            else:
                X_scaled = X_selected
            y_pred = classifier.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)

        print(f"\nAccuracy: {accuracy * 100:.2f}%")

        if do_cv:
            print("\nClassification Report:\n")
            print(classification_report(y, y_pred, target_names=class_names, zero_division=0))
        else:
            print("\nClassification Report:\n")
            print(classification_report(y, y_pred, target_names=class_names, zero_division=0))

        cm = confusion_matrix(y, y_pred)

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{clf_file} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        # improve tick label visibility
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(features_dir / f"{clf_file}_confusion_matrix.png", bbox_inches="tight")
        plt.close()

        # Warn about suspicious perfect accuracy
        if accuracy == 1.0:
            print(
                "WARNING: classifier achieved 100% accuracy on the evaluated set. This can indicate"
            )
            print(
                "  that the classifier was trained on the same data it is being evaluated on (data leakage) or"
            )
            print(
                "  severe overfitting. Consider evaluating on a held-out test set or using cross-validation."
            )

        results.append({
            "Classifier": save_object["classifier_name"],
            "Accuracy": accuracy * 100,
            "Search Policy": save_object.get("search_policy"),
            "Feature Count": len(selected_features)
        })


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved classifiers using features in the repo.")
    parser.add_argument("--features-dir", type=str, default=None, help="Override features directory")
    parser.add_argument("--cv", type=int, default=None, help="If set, run stratified N-fold cross-validation instead of using saved scaler/model (no leakage).")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    features_dir = Path(args.features_dir) if args.features_dir else find_features_dir(ROOT)

    classifier_files = [
        "NB_model.pkl",
        "LDA_model.pkl",
        "QDA_model.pkl",
        "LSVM_model.pkl",
        "DT_model.pkl",
        "KNN_model.pkl"
    ]

    results = []
    evaluate_on_dataset(features_dir, classifier_files, results, do_cv=args.cv)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("CLASSIFIER COMPARISON")
    print("=" * 70)
    print(results_df)

    results_df.to_csv(features_dir / "classifier_comparison.csv", index=False)
    print("\nSaved comparison CSV successfully.")


if __name__ == "__main__":
    main()
