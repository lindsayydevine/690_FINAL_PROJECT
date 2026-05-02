import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import load_preprocessed_h5
from scripts.generation import extract_tokens
from src.models.biopm import load_pretrained_encoder


real_path = "./GenData"
fake_path = "./synthetic_data/noise=0.03/Synthetic_MeLabel_P017.h5"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 64


def load_fake_tokens_and_labels(path):
    with h5py.File(path, "r") as f:
        print("Fake keys:", list(f.keys()))
        X = f["synthetic_tokens"][:]

        if "window_label" in f:
            y = f["window_label"][:]
        elif "source_labels" in f:
            y = f["source_labels"][:]
        else:
            raise KeyError("No label dataset found in fake file.")

    return X, y.astype(int)


def token_features(X):
    """
    X: [N, 192, 64]
    returns: [N, 128]
    """
    mean_feat = X.mean(axis=1)
    std_feat = X.std(axis=1)
    return np.concatenate([mean_feat, std_feat], axis=1)


# 1. Load real preprocessed IMU data
X, pos_info, add_emb, labels, pids, X_grav, raw_acc = load_preprocessed_h5(real_path)
y_real = labels.astype(int)

# 2. Extract real Bio-PM tokens
biopm_chkpt = "./checkpoints/checkpoint.pt"
print("Loading BioPM checkpoint...")
model_biopm = load_pretrained_encoder(biopm_chkpt, device=device)
model_biopm.eval()

real_tokens = extract_tokens(
    model_biopm,
    X,
    pos_info,
    add_emb,
    device=device,
    batch_size=batch_size,
)

real_tokens = real_tokens.cpu().numpy()

# 3. Load synthetic tokens
fake_tokens, y_fake = load_fake_tokens_and_labels(fake_path)

# 4. Convert token sequences into classifier features
X_real_feat = token_features(real_tokens)
X_fake_feat = token_features(fake_tokens)

print("Real feature shape:", X_real_feat.shape)
print("Fake feature shape:", X_fake_feat.shape)

# 5. Train classifier on real token features
X_train, X_test, y_train, y_test = train_test_split(
    X_real_feat,
    y_real,
    test_size=0.2,
    random_state=42,
    stratify=y_real,
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

clf.fit(X_train, y_train)

# 6. Real test performance
real_pred = clf.predict(X_test)

print("Real test accuracy:", accuracy_score(y_test, real_pred))
print(classification_report(y_test, real_pred, zero_division=0))

# 7. Fake label consistency
fake_pred = clf.predict(X_fake_feat)

print("Fake label consistency accuracy:", accuracy_score(y_fake, fake_pred))
print(classification_report(y_fake, fake_pred, zero_division=0))