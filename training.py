import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from model import load_dataset_cnn, CNN, cross_validate_model, Logger, save_model

# Configs
DATASET_DIR = 'recordings'
SEED = 42
KFOLD_SPLITS = 5
EPOCHS = 10


# Utils
def random_search_hyperparams(CNNClass, X, y, label_names, input_shape, k=5, n_iter=10, seed=42, logger=None):
    np.random.seed(seed)
    best_score = -np.inf
    best_model = None
    best_config = {}

    for i in range(n_iter):
        loggercv = Logger(f"logs/kfoldcv/iter_{i+1}_random_search_log.txt")
        # Randomly sample hyperparameters
        learning_rate = 10 ** np.random.uniform(-3, -1)  # e.g., from 0.001 to 0.1
        num_filters = np.random.choice([2, 4, 8, 16])

        iteration_msg = f"\n🔍 Random Search Iteration {i+1}: lr={learning_rate:.5f}, filters={num_filters}"
        print(iteration_msg)
        if logger:
            logger.log(iteration_msg)

        # Define model factory function for cross-validation
        def model_fn():
            return CNNClass(
                input_shape=input_shape,
                num_classes=len(label_names),
                learning_rate=learning_rate,
                num_filters=num_filters
            )

        # Cross-validation
        model, fold = cross_validate_model(model_fn, X, y, k=k, logger=loggercv)

        # Evaluate on training set (cross_val should handle this internally)
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        f1_msg = f"🔎 F1 Score (macro): {f1:.4f}"
        print(f1_msg)
        if logger:
            logger.log(f1_msg)

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_config = {
                'learning_rate': learning_rate,
                'num_filters': num_filters
            }
            best_msg = f"✨ New Best: F1 Score = {best_score:.4f}, Params = {best_config}"
            print(best_msg)
            if logger:
                logger.log(best_msg)

    final_msg = f"\n🏆 Best Hyperparameters: {best_config} with F1 Score: {best_score:.4f}"
    print(final_msg)
    if logger:
        logger.log(final_msg)

    return best_model, best_config


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


# Pipeline
def main():
    set_seed(SEED)

    print("📁 Loading dataset...")
    X, y = load_dataset_cnn(DATASET_DIR)
    label_names = [str(i) for i in sorted(set(y))]

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    print("🔁 Performing Random Search with Cross Validation...")
    logger = Logger("logs/random_search_log.txt")
    best_model, best_params = random_search_hyperparams(
        CNN, X_trainval, y_trainval, label_names,
        input_shape=X.shape[1:], k=KFOLD_SPLITS, n_iter=10, seed=SEED,
        logger=logger
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


if __name__ == '__main__':
    main()