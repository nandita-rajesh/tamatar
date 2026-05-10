import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

# =========================================================
# LOAD DATA
# =========================================================

features_dir = Path("features/tamatar")

X_raw = np.load(features_dir / "combined_features.npy")
y_raw = np.load(features_dir / "labels.npy")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# =========================================================
# CLASSIFIER FILES
# =========================================================

classifier_files = [
    "NB_model.pkl",
    "LDA_model.pkl",
    "QDA_model.pkl",
    "LSVM_model.pkl",
    "DT_model.pkl",
    "KNN_model.pkl"
]

results = []

# =========================================================
# EVALUATE EACH CLASSIFIER
# =========================================================

for clf_file in classifier_files:

    print("\n" + "=" * 70)
    print(f"Evaluating {clf_file}")
    print("=" * 70)

    with open(features_dir / clf_file, "rb") as f:
        save_object = pickle.load(f)

    classifier = save_object["model"]
    scaler = save_object["scaler"]
    selected_features = save_object["selected_features"]
    class_names = [str(c) for c in save_object["label_classes"]]

    # Select saved features
    X_selected = X_raw[:, selected_features]

    # Scale
    X_scaled = scaler.transform(X_selected)

    # Predict
    y_pred = classifier.predict(X_scaled)

    # Accuracy
    accuracy = accuracy_score(y, y_pred)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("\nClassification Report:\n")

    print(
        classification_report(
            y,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"{clf_file} Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        features_dir / f"{clf_file}_confusion_matrix.png"
    )

    plt.close()

    # Store results
    results.append({
        "Classifier": save_object["classifier_name"],
        "Accuracy": accuracy * 100,
        "Search Policy": save_object["search_policy"],
        "Feature Count": len(selected_features)
    })

# =========================================================
# FINAL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("CLASSIFIER COMPARISON")
print("=" * 70)

print(results_df)

results_df.to_csv(
    features_dir / "classifier_comparison.csv",
    index=False
)

print("\nSaved comparison CSV successfully.")