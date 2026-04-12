"""
Matplotlib metrics for the Flood (binary classification) model:
- Confusion matrix
- Training vs validation accuracy (learning curve)
- Training vs validation log loss (learning curve)
- F1 score (macro / weighted / per-class bar chart)

Run from project folder: python visualize_flood_metrics.py
Outputs PNGs under ./plots/
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "plots")
os.makedirs(OUT, exist_ok=True)


def build_flood_data(seed: int = 42, n: int = 2000):
    np.random.seed(seed)
    df = pd.DataFrame({
        "Latitude": np.random.uniform(8, 37, n),
        "Longitude": np.random.uniform(68, 97, n),
        "Rainfall (mm)": np.random.uniform(0, 300, n),
        "Temperature (°C)": np.random.uniform(15, 45, n),
        "Humidity (%)": np.random.uniform(20, 100, n),
        "River Discharge (m³/s)": np.random.uniform(0, 5000, n),
        "Water Level (m)": np.random.uniform(0, 10, n),
        "Elevation (m)": np.random.uniform(1, 9000, n),
        "Land Cover": np.random.choice(
            ["Agricultural", "Desert", "Forest", "Urban", "Water Body"], n
        ),
        "Soil Type": np.random.choice(["Clay", "Loam", "Peat", "Sandy", "Silt"], n),
        "Population Density": np.random.uniform(2, 10000, n),
        "Infrastructure": np.random.choice([0, 1], n),
        "Historical Floods": np.random.choice([0, 1], n),
    })
    flood_risk = (df["Rainfall (mm)"] > 150) & (df["Water Level (m)"] > 5)
    df["Flood Occurred"] = (
        flood_risk.astype(int) + np.random.choice([0, 1], n, p=[0.3, 0.7])
    ).clip(0, 1)
    X = df.drop("Flood Occurred", axis=1)
    y = df["Flood Occurred"]
    X_enc = pd.get_dummies(X, drop_first=True)
    return X_enc, y


def main():
    X, y = build_flood_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # --- Confusion matrix ---
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No flood", "Flood"], cmap="Blues", ax=ax_cm
    )
    ax_cm.set_title("Flood model — confusion matrix (test set)")
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
    plt.close(fig_cm)

    # --- F1 scores ---
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    labels = [0, 1]
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=labels)
    names = ["No flood (0)", "Flood (1)"]

    fig_f1, ax_f1 = plt.subplots(figsize=(7, 4))
    xpos = np.arange(len(names))
    ax_f1.bar(xpos, f1_per_class, color=["#4472c4", "#ed7d31"], width=0.55)
    ax_f1.set_xticks(xpos)
    ax_f1.set_xticklabels(names)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_ylabel("F1 score")
    ax_f1.set_title(
        f"F1 scores — macro={f1_macro:.3f}, weighted={f1_weighted:.3f}"
    )
    for i, v in enumerate(f1_per_class):
        ax_f1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    fig_f1.tight_layout()
    fig_f1.savefig(os.path.join(OUT, "f1_scores.png"), dpi=150)
    plt.close(fig_f1)

    # Learning curves (train vs validation) — same random splits as typical "curves"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 8)

    # Accuracy
    train_sizes_abs, train_acc, val_acc = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )
    train_acc_mean = np.mean(train_acc, axis=1)
    val_acc_mean = np.mean(val_acc, axis=1)
    train_acc_std = np.std(train_acc, axis=1)
    val_acc_std = np.std(val_acc, axis=1)

    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(
        train_sizes_abs, train_acc_mean, "o-", color="#1f77b4", label="Training accuracy"
    )
    ax_acc.fill_between(
        train_sizes_abs,
        train_acc_mean - train_acc_std,
        train_acc_mean + train_acc_std,
        alpha=0.15,
        color="#1f77b4",
    )
    ax_acc.plot(
        train_sizes_abs,
        val_acc_mean,
        "o-",
        color="#ff7f0e",
        label="Validation accuracy",
    )
    ax_acc.fill_between(
        train_sizes_abs,
        val_acc_mean - val_acc_std,
        val_acc_mean + val_acc_std,
        alpha=0.15,
        color="#ff7f0e",
    )
    ax_acc.set_xlabel("Training examples")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy — training vs validation (learning curve)")
    ax_acc.legend(loc="best")
    ax_acc.grid(True, alpha=0.3)
    fig_acc.tight_layout()
    fig_acc.savefig(os.path.join(OUT, "accuracy_train_val.png"), dpi=150)
    plt.close(fig_acc)

    # Log loss (lower is better); sklearn returns neg_log_loss
    train_sizes_ll, train_nll, val_nll = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=1,
    )
    train_loss = -np.mean(train_nll, axis=1)
    val_loss = -np.mean(val_nll, axis=1)
    train_loss_std = np.std(-train_nll, axis=1)
    val_loss_std = np.std(-val_nll, axis=1)

    fig_ll, ax_ll = plt.subplots(figsize=(8, 5))
    ax_ll.plot(
        train_sizes_ll, train_loss, "o-", color="#2ca02c", label="Training log loss"
    )
    ax_ll.fill_between(
        train_sizes_ll,
        train_loss - train_loss_std,
        train_loss + train_loss_std,
        alpha=0.15,
        color="#2ca02c",
    )
    ax_ll.plot(
        train_sizes_ll,
        val_loss,
        "o-",
        color="#d62728",
        label="Validation log loss",
    )
    ax_ll.fill_between(
        train_sizes_ll,
        val_loss - val_loss_std,
        val_loss + val_loss_std,
        alpha=0.15,
        color="#d62728",
    )
    ax_ll.set_xlabel("Training examples")
    ax_ll.set_ylabel("Log loss")
    ax_ll.set_title("Log loss — training vs validation (learning curve)")
    ax_ll.legend(loc="best")
    ax_ll.grid(True, alpha=0.3)
    fig_ll.tight_layout()
    fig_ll.savefig(os.path.join(OUT, "loss_train_val.png"), dpi=150)
    plt.close(fig_ll)

    # Combined summary figure
    fig_all, axes = plt.subplots(2, 2, figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No flood", "Flood"], cmap="Blues", ax=axes[0, 0]
    )
    axes[0, 0].set_title("Confusion matrix (test)")

    axes[0, 1].bar(xpos, f1_per_class, color=["#4472c4", "#ed7d31"], width=0.55)
    axes[0, 1].set_xticks(xpos)
    axes[0, 1].set_xticklabels(names, rotation=15, ha="right")
    axes[0, 1].set_ylabel("F1")
    axes[0, 1].set_title(f"F1 (macro={f1_macro:.3f}, weighted={f1_weighted:.3f})")
    axes[0, 1].set_ylim(0, 1.05)

    axes[1, 0].plot(train_sizes_abs, train_acc_mean, "o-", label="Train acc", color="#1f77b4")
    axes[1, 0].plot(train_sizes_abs, val_acc_mean, "o-", label="Val acc", color="#ff7f0e")
    axes[1, 0].set_xlabel("Training examples")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy: train vs validation")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(train_sizes_ll, train_loss, "o-", label="Train log loss", color="#2ca02c")
    axes[1, 1].plot(train_sizes_ll, val_loss, "o-", label="Val log loss", color="#d62728")
    axes[1, 1].set_xlabel("Training examples")
    axes[1, 1].set_ylabel("Log loss")
    axes[1, 1].set_title("Log loss: train vs validation")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig_all.suptitle("Flood classifier — evaluation summary", fontsize=12, y=1.02)
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(OUT, "flood_metrics_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_all)

    print(f"Saved plots to: {OUT}")
    print("  - confusion_matrix.png")
    print("  - f1_scores.png")
    print("  - accuracy_train_val.png")
    print("  - loss_train_val.png")
    print("  - flood_metrics_summary.png")


if __name__ == "__main__":
    main()
