import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from model import load_dataset_cnn, CNN, cross_validate_model, Logger, save_model

# ------------------------
# CONFIGURATION
# ------------------------
DATASET_DIR = 'recordings'
SEED = 42
KFOLD_SPLITS = 5
EPOCHS = 10


# ------------------------
# UTILITIES
# ------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X, y, label_names):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred, average=None, zero_division=0)
    recall = recall_score(y, y_pred, average=None, zero_division=0)
    f1 = f1_score(y, y_pred, average=None, zero_division=0)

    accuracy = np.mean(y_pred == y)
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_avg = np.mean(f1)

    print(f"🎯 Accuracy: {accuracy:.2%}, Precision: {precision_avg:.2%}, Recall: {recall_avg:.2%}, F1: {f1_avg:.2%}")
    plot_confusion_matrix(cm, label_names)
    return accuracy, precision_avg, recall_avg, f1_avg


# ------------------------
# MAIN PIPELINE
# ------------------------
def main():
    set_seed(SEED)

    print("📁 Loading dataset...")
    X, y = load_dataset_cnn(DATASET_DIR)
    label_names = [str(i) for i in sorted(set(y))]

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    print("🔁 Performing K-Fold Cross Validation on training set...")
    best_model, best_fold = cross_validate_model(
        CNN, X_trainval, y_trainval,
        k=10,
        input_shape=X.shape[1:], num_classes=len(label_names), learning_rate=0.01
    )

    # Step 2: Save the best model and testing
    accuracy, precision, recall, f1 = evaluate_model(best_model, X_test, y_test, label_names)
    save_model(
        best_model,
        f"model/cnn_speech_model.pkl",
        label_names=label_names,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mean=np.mean(X_trainval) if isinstance(X_trainval, np.ndarray) else None,
        std=np.std(X_trainval) if isinstance(X_trainval, np.ndarray) else None
    )


# ------------------------
# ENTRY POINT
# ------------------------
if __name__ == '__main__':
    main()