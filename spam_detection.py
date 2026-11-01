"""
Spam Detection Framework using Machine Learning
Uses the supplied features.txt dataset (six numeric account/tweet features + class).

Expected columns:
Replies, Behaviour, Following, Followers, Burstiness, NegativeRatio, class

Run:
    pip install pandas numpy scikit-learn matplotlib
    python spam_detection.py

Important:
- This program evaluates the labels already present in features.txt.
- It does not prove those labels are correct. Verify how the original class labels
  were assigned before reporting model performance.
- With only 36 rows, test metrics are highly unstable. Treat them as a demonstration.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

FEATURE_COLUMNS = [
    "Replies", "Behaviour", "Following",
    "Followers", "Burstiness", "NegativeRatio"
]
TARGET_COLUMN = "class"

MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", class_weight="balanced", random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=3))
    ]),
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),
}


class SpamDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Detection Framework using Machine Learning")
        self.root.geometry("1050x720")

        self.data = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.results = {}
        self.trained_models = {}

        tk.Label(
            root, text="Spam Detection Framework using Machine Learning",
            font=("Arial", 17, "bold")
        ).pack(pady=12)

        controls = tk.Frame(root)
        controls.pack(pady=8)

        tk.Button(
            controls, text="Upload features.txt / CSV",
            command=self.load_dataset, width=23
        ).grid(row=0, column=0, padx=5, pady=5)

        tk.Button(
            controls, text="Prepare Train/Test Data",
            command=self.prepare_data, width=23
        ).grid(row=0, column=1, padx=5, pady=5)

        tk.Button(
            controls, text="Train & Evaluate All Models",
            command=self.train_models, width=25
        ).grid(row=0, column=2, padx=5, pady=5)

        tk.Button(
            controls, text="Show Comparison Graph",
            command=self.show_graph, width=23
        ).grid(row=0, column=3, padx=5, pady=5)

        self.status = tk.Label(root, text="Upload the feature dataset to begin.")
        self.status.pack(pady=4)

        self.output = tk.Text(root, wrap="word", font=("Courier New", 10))
        self.output.pack(fill="both", expand=True, padx=12, pady=10)

    def log(self, message):
        self.output.insert("end", message + "\n")
        self.output.see("end")

    def load_dataset(self):
        path = filedialog.askopenfilename(
            title="Select feature dataset",
            filetypes=[("Text/CSV files", "*.txt *.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            # The supplied features.txt is comma-separated with a header row.
            df = pd.read_csv(path, sep=",")
            df.columns = df.columns.astype(str).str.strip()

            missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN]
                       if c not in df.columns]
            if missing:
                raise ValueError(
                    "Missing required columns: " + ", ".join(missing) +
                    "\nExpected: " + ", ".join(FEATURE_COLUMNS + [TARGET_COLUMN])
                )

            df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
            for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna()
            df = df.drop_duplicates().reset_index(drop=True)

            if len(df) < 10:
                raise ValueError("The dataset has too few valid rows for a train/test split.")

            labels = set(df[TARGET_COLUMN].unique())
            if not labels.issubset({0, 1}):
                raise ValueError(
                    f"The class column must contain only 0 and 1. Found: {sorted(labels)}"
                )
            if df[TARGET_COLUMN].nunique() != 2:
                raise ValueError("Both classes (0 and 1) must be present.")

            self.data = df
            self.X_train = self.X_test = None
            self.results.clear()
            self.trained_models.clear()
            self.output.delete("1.0", "end")

            self.status.config(text=f"Loaded: {Path(path).name} | {len(df)} valid rows")
            self.log(f"Dataset loaded: {path}")
            self.log(f"Rows after cleaning: {len(df)}")
            self.log(f"Class distribution:\n{df[TARGET_COLUMN].value_counts().sort_index()}")
            self.log("\nFeature columns: " + ", ".join(FEATURE_COLUMNS))
            self.log("\nNote: class labels are taken directly from the supplied file.")
        except Exception as exc:
            messagebox.showerror("Dataset error", str(exc))

    def prepare_data(self):
        if self.data is None:
            messagebox.showwarning("No dataset", "Upload the dataset first.")
            return

        X = self.data[FEATURE_COLUMNS]
        y = self.data[TARGET_COLUMN].astype(int)

        if y.value_counts().min() < 2:
            messagebox.showerror(
                "Insufficient class samples",
                "Each class needs at least two rows for a stratified split."
            )
            return

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.results.clear()
        self.trained_models.clear()

        self.output.delete("1.0", "end")
        self.log(f"Total valid rows: {len(self.data)}")
        self.log(f"Training rows: {len(self.X_train)}")
        self.log(f"Testing rows: {len(self.X_test)}")
        self.log("\nStratified split completed. Test data is kept separate from training data.")

    def train_models(self):
        if self.X_train is None:
            self.prepare_data()
            if self.X_train is None:
                return

        self.results.clear()
        self.trained_models.clear()

        for name, model in MODELS.items():
            try:
                model.fit(self.X_train, self.y_train)
                predictions = model.predict(self.X_test)

                metrics = {
                    "Accuracy": accuracy_score(self.y_test, predictions) * 100,
                    "Precision": precision_score(
                        self.y_test, predictions, zero_division=0
                    ) * 100,
                    "Recall": recall_score(
                        self.y_test, predictions, zero_division=0
                    ) * 100,
                    "F1-score": f1_score(
                        self.y_test, predictions, zero_division=0
                    ) * 100,
                }
                self.results[name] = metrics
                self.trained_models[name] = model

                self.log(f"\n{'=' * 55}\n{name}")
                for metric, value in metrics.items():
                    self.log(f"{metric:12s}: {value:.2f}%")
                self.log("\nClassification report:")
                self.log(classification_report(
                    self.y_test, predictions, labels=[0, 1],
                    target_names=["Non-Spam (0)", "Spam (1)"],
                    zero_division=0
                ))
                self.log("Confusion matrix:")
                self.log(str(confusion_matrix(self.y_test, predictions, labels=[0, 1])))

            except Exception as exc:
                self.log(f"\n{name} failed: {exc}")

        if self.results:
            self.log(
                "\nIMPORTANT: These are results from a small test split, "
                "not proof of general performance."
            )
            self.status.config(text="Model training and evaluation completed.")

    def show_graph(self):
        if not self.results:
            messagebox.showwarning("No results", "Train the models first.")
            return

        metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
        names = list(self.results.keys())
        x = np.arange(len(names))
        width = 0.18

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in names]
            ax.bar(x + (i - 1.5) * width, values, width, label=metric)

        ax.set_title("Classifier Performance Comparison")
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectionApp(root)
    root.mainloop()
