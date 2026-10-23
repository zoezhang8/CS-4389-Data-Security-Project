"""
Data poisoning track: label flipping against the team's victim model.

Uses the shared pieces from victim_train.py so results are comparable with
the baseline (same seeded split, same VictimNet, same training settings):
    get_data()     -> victim_train, attacker_pool, validation, test
    train_model()  -> trains a fresh VictimNet on any (image, label) dataset
    evaluate()     -> accuracy on a dataset

Threat model: the attacker can change the LABELS of part of the victim's
training set (victim_train), but not the images and not the test set.

Experiments
  1. random   : flip `rate` of all training labels to a random wrong class
  2. targeted : flip `rate` of the SOURCE-class labels to TARGET (e.g. 7 -> 1)
Defense preview
  k-NN label-consistency filter: drop training samples whose label disagrees
  with the majority label of their k nearest neighbors (raw pixels), then
  retrain. Reports detection TPR / FPR.

Run from this folder:   python label_flip.py
Outputs go to ./poison_results/ (results.csv, label_flip_mnist.png)

Imports victim_train from the shared ../ai_security_project_victim/ module
(same one fahad_api/api.py points at via VICTIM_DIR) instead of a local
copy, so every track trains/evaluates against the exact same victim code.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ai_security_project_victim"))
import victim_train as vt

FLIP_RATES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
SOURCE, TARGET = 7, 1          # targeted attack: relabel some 7s as 1s
KNN_K = 10
EPOCHS = 5                     # same as baseline
OUT_DIR = os.path.join(vt.HERE, "poison_results")


# ---------------------------------------------------------------- datasets
class RelabeledDataset(Dataset):
    """Wraps a dataset and serves (image, new_label[i]) instead of the original label.

    Only labels change; images are read from the original dataset, so
    nothing is copied in memory.
    """

    def __init__(self, base, labels, keep=None):
        self.base = base
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.keep = np.arange(len(base)) if keep is None else np.asarray(keep)

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        j = int(self.keep[i])
        x, _ = self.base[j]
        return x, self.labels[j]


def get_labels(subset):
    """Original labels of a random_split Subset without loading images."""
    return subset.dataset.targets[subset.indices].numpy().copy()


def get_images(subset):
    """All images of a Subset as an (N, 784) float array (used by the k-NN filter)."""
    return subset.dataset.data[subset.indices].reshape(len(subset), -1).numpy() / 255.0


# ---------------------------------------------------------------- attacks
def flip_random(y, rate, rng, num_classes=10):
    """Flip `rate` of all labels to a random different class."""
    y_p = y.copy()
    idx = rng.choice(len(y), int(rate * len(y)), replace=False)
    for i in idx:
        y_p[i] = rng.choice([c for c in range(num_classes) if c != y[i]])
    return y_p, idx


def flip_targeted(y, rate, rng, source=SOURCE, target=TARGET):
    """Flip `rate` of the source-class labels to the target class."""
    y_p = y.copy()
    src = np.where(y == source)[0]
    idx = rng.choice(src, int(rate * len(src)), replace=False)
    y_p[idx] = target
    return y_p, idx


# ---------------------------------------------------------------- defense
def knn_filter(X, y, k=KNN_K):
    """Return indices whose label disagrees with the majority of their k neighbors."""
    nn_index = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neigh = nn_index.kneighbors(X, return_distance=False)[:, 1:]  # drop self
    majority = np.array([np.bincount(y[n], minlength=10).argmax() for n in neigh])
    return np.where(majority != y)[0]


def detection_rates(flagged, poisoned, n):
    flagged, poisoned = set(flagged.tolist()), set(poisoned.tolist())
    if not poisoned:
        return float("nan"), len(flagged) / n
    tp = len(flagged & poisoned)
    return tp / len(poisoned), (len(flagged) - tp) / (n - len(poisoned))


# ---------------------------------------------------------------- evaluation
def class_recall(model, dataset, cls):
    """Fraction of test samples of class `cls` that the model predicts correctly."""
    model.eval()
    hit = total = 0
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=256):
            mask = y == cls
            if mask.any():
                hit += (model(x[mask]).argmax(1) == cls).sum().item()
                total += mask.sum().item()
    return hit / total


def run_one(train_base, labels, keep, test):
    model = vt.train_model(RelabeledDataset(train_base, labels, keep),
                           epochs=EPOCHS, verbose=False)
    return vt.evaluate(model, test), class_recall(model, test, SOURCE)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    victim_train, _attacker_pool, _validation, test = vt.get_data()
    y_clean = get_labels(victim_train)
    X = get_images(victim_train)
    print(f"victim train: {len(y_clean)}  test: {len(test)}")

    rows = []
    for attack in ["random", "targeted"]:
        for rate in FLIP_RATES:
            rng = np.random.default_rng(vt.SEED)
            if attack == "random":
                y_p, poisoned = flip_random(y_clean, rate, rng)
            else:
                y_p, poisoned = flip_targeted(y_clean, rate, rng)

            acc, rec = run_one(victim_train, y_p, None, test)

            flagged = knn_filter(X, y_p)
            tpr, fpr = detection_rates(flagged, poisoned, len(y_p))
            keep = np.setdiff1d(np.arange(len(y_p)), flagged)
            acc_d, rec_d = run_one(victim_train, y_p, keep, test)

            row = {"attack": attack, "flip_rate": rate, "n_flipped": len(poisoned),
                   "test_acc": acc, f"class{SOURCE}_recall": rec,
                   "test_acc_defended": acc_d, f"class{SOURCE}_recall_defended": rec_d,
                   "detect_TPR": tpr, "detect_FPR": fpr, "n_removed": len(flagged)}
            rows.append(row)
            print(f"{attack:8s} rate={rate:.2f} flipped={len(poisoned):5d} | "
                  f"acc={acc:.4f} class{SOURCE}_recall={rec:.4f} | "
                  f"defended acc={acc_d:.4f} recall={rec_d:.4f} | "
                  f"TPR={tpr:.2f} FPR={fpr:.3f}")

    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    pct = [r * 100 for r in FLIP_RATES]
    panels = [(axes[0], "random", "test_acc", "Random flipping: test accuracy"),
              (axes[1], "targeted", f"class{SOURCE}_recall",
               f"Targeted flipping ({SOURCE}->{TARGET}): class-{SOURCE} recall")]
    for ax, attack, metric, title in panels:
        sub = [r for r in rows if r["attack"] == attack]
        ax.plot(pct, [r[metric] for r in sub], "o-", color="#c0392b", label="No defense")
        ax.plot(pct, [r[metric + "_defended"] for r in sub], "s--", color="#2471a3",
                label="k-NN filter")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Flip rate (%)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("VictimNet (MNIST) under label-flipping poisoning", fontsize=12)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "label_flip_mnist.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
