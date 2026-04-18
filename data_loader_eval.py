"""data loading and evaluation for tool wear prediction."""

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from pathlib import Path
import pandas as pd


def load_data(window_size=64, stride=64):
    """Load and split data into train/val/test."""
    # Load raw data
    data = sio.loadmat('data/raw/dataset_a.mat', struct_as_record=True)['mill']
    df_labels = pd.read_csv('data/processed/labels_with_tool_class.csv')
    df_labels.drop([17, 94], inplace=True, errors='ignore')  # Remove bad cuts

    signal_names = data.dtype.names[7:]  # Signal columns
    X_list, y_list = [], []

    for _, row in df_labels.iterrows():
        cut_no, tool_class = int(row['cut_no']), int(row['tool_class'])
        if cut_no >= data.shape[1]:
            continue

        # Stack all signals for this cut
        signals = np.column_stack([data[0, cut_no][s].flatten() for s in signal_names])

        # Create windows
        for start in range(0, len(signals) - window_size + 1, stride):
            X_list.append(signals[start:start + window_size])
            y_list.append(tool_class)

    X, y = np.array(X_list), np.array(y_list)

    # Split: train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.33, random_state=15, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=10, stratify=y_temp)

    # Min-max scale using training stats
    min_v, max_v = X_train.min(axis=(0,1)), X_train.max(axis=(0,1))
    X_train = (X_train - min_v) / (max_v - min_v + 1e-8)
    X_val = (X_val - min_v) / (max_v - min_v + 1e-8)
    X_test = (X_test - min_v) / (max_v - min_v + 1e-8)

    return X_train, X_val, X_test, y_train, y_val, y_test


def random_predict(y_true, seed=42):
    """Generate random predictions mimicking ML model."""
    np.random.seed(seed)
    n = len(y_true)
    y_pred = np.random.randint(0, 3, n)  # 3 classes
    y_prob = np.random.rand(n)  # Probability for binary eval
    return y_pred, y_prob


def evaluate(y_true, y_pred, y_prob):
    """Evaluate predictions with metrics."""
    print("\n=== MULTICLASS EVALUATION ===")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Degraded', 'Failed']))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # Binary: healthy (0) vs worn (1,2)
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    print("\n=== BINARY EVALUATION (Healthy vs Worn) ===")
    print(classification_report(y_true_bin, y_pred_bin, target_names=['Healthy', 'Worn']))

    fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
    print(f"ROC AUC: {auc(fpr, tpr):.4f}")


def main():
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Classes in test: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    print("\nGenerating random predictions...")
    y_pred, y_prob = random_predict(y_test)
    evaluate(y_test, y_pred, y_prob)


if __name__ == "__main__":
    main()

