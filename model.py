import os
import numpy as np
import random
from scipy.io import wavfile
from scipy.fftpack import dct
from sklearn.model_selection import KFold
from tqdm import tqdm

class Logger:
    def __init__(self, filename="../training_log.txt"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write("Training Log\n")

    def log(self, message):
        with open(self.filename, 'a') as f:
            f.write(message + "\n")


def extract_mfcc_matrix(signal, sr, n_mfcc=20):
    signal = signal / np.max(np.abs(signal))
    frame_size = int(0.025 * sr)
    frame_step = int(0.01 * sr)
    signal_len = len(signal)
    num_frames = int(np.ceil((signal_len - frame_size) / frame_step)) + 1

    pad_len = (num_frames - 1) * frame_step + frame_size
    pad_signal = np.append(signal, np.zeros(pad_len - signal_len))

    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T
    frames = pad_signal[indices] * np.hamming(frame_size)

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + sr / 2 / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, 28)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((26, int(NFFT / 2 + 1)))
    for m in range(1, 27):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_fbank = np.log(filter_banks)

    mfccs = dct(log_fbank, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return mfccs.T


def load_dataset_cnn(directory='recordings', sr=8000):
    X, y = [], []
    for fname in os.listdir(directory):
        if fname.endswith('.wav'):
            label = int(fname[0])
            sr_read, signal = wavfile.read(os.path.join(directory, fname))
            if sr_read != sr:
                continue
            mfcc = extract_mfcc_matrix(signal, sr)
            if mfcc.shape[1] < 16:
                continue
            X.append(mfcc[:, :16])
            y.append(label)
    X = np.expand_dims(np.array(X), 1)
    return X, np.array(y)


class CNN:
    def __init__(self, input_shape=(1, 20, 16), num_classes=10, learning_rate=0.01):
        self.lr = learning_rate
        self.num_classes = num_classes

        # Initialize weights
        self.conv_filters = np.random.randn(4, 1, 3, 3) * 0.1  # (out_channels, in_channels, kH, kW)
        self.fc_weights = np.random.randn(252, num_classes) * 0.1
        self.fc_bias = np.zeros(num_classes)

    def relu(self, x):
        self.relu_mask = (x > 0)
        return x * self.relu_mask

    def relu_backward(self, grad_output):
        return grad_output * self.relu_mask

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def cross_entropy_loss(self, probs, labels):
        m = labels.shape[0]
        log_likelihood = -np.log(probs[np.arange(m), labels] + 1e-15)
        loss = np.sum(log_likelihood) / m
        return loss

    def cross_entropy_grad(self, probs, labels):
        m = labels.shape[0]
        grad = probs.copy()
        grad[np.arange(m), labels] -= 1
        grad /= m
        return grad

    def conv_forward(self, x):
        self.x = x  # Save input
        batch_size, _, h, w = x.shape
        out = np.zeros((batch_size, 4, h - 2, w - 2))

        for b in range(batch_size):
            for f in range(4):
                for i in range(h - 2):
                    for j in range(w - 2):
                        region = x[b, 0, i:i+3, j:j+3]
                        out[b, f, i, j] = np.sum(region * self.conv_filters[f, 0])
        return out

    def max_pool_forward(self, x):
        self.pool_input = x
        batch_size, channels, h, w = x.shape
        out = np.zeros((batch_size, channels, h // 2, w // 2))
        self.pool_mask = np.zeros_like(x)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, h, 2):
                    for j in range(0, w, 2):
                        patch = x[b, c, i:i+2, j:j+2]
                        max_val = np.max(patch)
                        out[b, c, i//2, j//2] = max_val
                        mask = (patch == max_val)
                        self.pool_mask[b, c, i:i+2, j:j+2] = mask
        return out

    def max_pool_backward(self, d_out):
        d_input = np.zeros_like(self.pool_input)
        b, c, h, w = d_out.shape

        for i in range(h):
            for j in range(w):
                d_input[:, :, i*2:i*2+2, j*2:j*2+2] += self.pool_mask[:, :, i*2:i*2+2, j*2:j*2+2] * d_out[:, :, i, j][:, :, None, None]
        return d_input

    def conv_backward(self, d_out):
        d_filters = np.zeros_like(self.conv_filters)
        b, _, h, w = self.x.shape

        for batch in range(b):
            for f in range(4):
                for i in range(h - 2):
                    for j in range(w - 2):
                        region = self.x[batch, 0, i:i+3, j:j+3]
                        d_filters[f, 0] += d_out[batch, f, i, j] * region
        self.conv_filters -= self.lr * d_filters

    def fc_forward(self, x):
        self.fc_input = x
        return np.dot(x, self.fc_weights) + self.fc_bias

    def fc_backward(self, d_logits):
        dW = np.dot(self.fc_input.T, d_logits)
        db = np.sum(d_logits, axis=0)
        dx = np.dot(d_logits, self.fc_weights.T)

        self.fc_weights -= self.lr * dW
        self.fc_bias -= self.lr * db

        return dx

    def forward(self, x):
        out = self.conv_forward(x)
        out = self.relu(out)
        out = self.max_pool_forward(out)
        out = out.reshape(out.shape[0], -1)
        logits = self.fc_forward(out)
        return logits

    def train_step(self, x, y):
        logits = self.forward(x)
        probs = self.softmax(logits)
        loss = self.cross_entropy_loss(probs, y)

        # Backprop
        d_logits = self.cross_entropy_grad(probs, y)
        d_fc = self.fc_backward(d_logits)
        d_fc_reshaped = d_fc.reshape(x.shape[0], 4, 9, 7)
        d_pool = self.max_pool_backward(d_fc_reshaped)
        d_relu = self.relu_backward(d_pool)
        self.conv_backward(d_relu)

        return loss

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(self.softmax(logits), axis=1)
    

def train(model, X_train, y_train, epochs=10, batch_size=32, logger=None):
    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0
        correct = 0
        num_batches = int(np.ceil(num_samples / batch_size))

        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                loss = model.train_step(X_batch, y_batch)
                preds = model.predict(X_batch)
                acc = np.mean(preds == y_batch)

                epoch_loss += loss * len(X_batch)
                correct += np.sum(preds == y_batch)

                pbar.set_postfix(loss=loss, acc=acc * 100)
                pbar.update(1)

        avg_loss = epoch_loss / num_samples
        avg_acc = correct / num_samples
        log_msg = f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Accuracy = {avg_acc*100:.2f}%"
        if logger:
            logger.log(log_msg)
        else:
            print(log_msg)


def cross_validate_model(model_class, X, y, k=5, **model_kwargs):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    best_model = None
    best_acc = 0
    best_fold = 0

    with tqdm(total=k, desc="Cross-Validation Folds", unit="fold") as fold_bar:
        for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
            logger = Logger(f"logs/log_fold_{fold}.txt")
            logger.log(f"=== Fold {fold}/{k} ===")

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = model_class(**model_kwargs)
            train(model, X_train, y_train, epochs=10, batch_size=32, logger=logger)

            preds = model.predict(X_val)
            acc = np.mean(preds == y_val)
            logger.log(f"Validation Accuracy for Fold {fold}: {acc * 100:.2f}%")
            fold_results.append(acc)

            # Track best model
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_fold = fold

            fold_bar.update(1)

    avg_acc = np.mean(fold_results)
    summary_logger = Logger("logs/log_crossval_summary.txt")
    summary_logger.log(f"Average Cross-Validation Accuracy: {avg_acc * 100:.2f}%")

    return best_model, best_fold


import pickle

def save_model(model, filepath, label_names=None, accuracy=None, precision=None,
               recall=None, f1=None, mean=None, std=None):
    model_data = {
        'model': model,
        'label_names': label_names,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean': mean,
        'std': std,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

def load_model():
    with open("model/cnn_speech_model.pkl", "rb") as f:
        model_data = pickle.load(f)
        return model_data  