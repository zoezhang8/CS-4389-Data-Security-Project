""" Zoe Zhang, 2024-06-05
Model extraction track: steal a copy of the victim model through its query API

Attack:
  1. query_victim() sends attacker_pool images to /predict and keeps the returned probability vectors as soft labels.
  2. A substitute CNN is trained on pairs for several query budgets. Budgets are nested
     prefixes of one shuffled order, so budget 20000 = the whole pool

run python extract.py
outputs go to ./extraction_results/ (results.csv, extraction_mnist.png,substitute_best.pt)
"""
import csv
import os
import sys
import time

os.environ["MODEL"] = "victim"   #api.py reads this at import: real model/ fail

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
#victim_train is only used for get_data() the victim's weights and VictimNet are reached only through the API
sys.path.insert(0, os.path.join(ROOT, "ai_security_project_victim"))
sys.path.insert(0, os.path.join(ROOT, "fahad_api"))
import victim_train as vt  # noqa: E402
import api  # noqa: E402

BUDGETS = [1000, 2500, 5000, 10000, 20000]
CLIENT_ID = "zoe-extraction"
EVAL_CLIENT_ID = "zoe-extraction-eval"
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
OUT_DIR = os.path.join(HERE, "extraction_results")


# API access
def query_victim(images, client_id=CLIENT_ID):
    """Send each (784,) image in [0, 1] to /predict; return (N, 10) probabilities.

    The API takes one image per request, so this is one logged query per image.
    """
    client = api.app.test_client()
    probs = np.empty((len(images), 10), dtype=np.float32)
    for i, x in enumerate(images):
        r = client.post("/predict", json={"image": x.tolist()},
                        headers={"X-Client-Id": client_id})
        if r.status_code != 200:
            raise RuntimeError(f"query {i} failed: {r.status_code} {r.get_json()}")
        probs[i] = r.get_json()["probabilities"]
        if (i + 1) % 5000 == 0:
            print(f"  {client_id}: {i + 1}/{len(images)} queries")
    return probs


# subsitute
class SubstituteNet(nn.Module):
    """Small CNN: 2x (conv5x5 -> ReLU -> maxpool) -> 128 -> 10.

    Black-box assumption: the attacker doesn't know the victim is a 784-128-64-10
    MLP. They only know the input format (28x28 grayscale digits, from the API
    docs), so they pick a generic, well-known architecture for that kind of
    data. A small CNN is the usual default for MNIST-sized images and has enough
    capacity to imitate an MLP's decision boundaries, so a high fidelity here
    shows extraction works without architecture knowledge.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 16x14x14
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def soft_cross_entropy(logits, target_probs):
    """Cross-entropy against the victim's full probability vector (distillation)."""
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def train_substitute(images, soft_labels, epochs=EPOCHS, seed=vt.SEED):
    torch.manual_seed(seed)
    model = SubstituteNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data = TensorDataset(torch.as_tensor(images).reshape(-1, 1, 28, 28),
                         torch.as_tensor(soft_labels))
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        model.train()
        for x, p in loader:
            optimizer.zero_grad()
            soft_cross_entropy(model(x), p).backward()
            optimizer.step()
    model.eval()
    return model


def predict(model, images, batch_size=1024):
    x = torch.as_tensor(images).reshape(-1, 1, 28, 28)
    with torch.no_grad():
        return torch.cat([model(b).argmax(1) for b in x.split(batch_size)]).numpy()


#data
def get_images(subset_or_dataset):
    """Images as an (N, 784) float32 array in [0, 1]; labels are not touched."""
    if hasattr(subset_or_dataset, "indices"):
        raw = subset_or_dataset.dataset.data[subset_or_dataset.indices]
    else:
        raw = subset_or_dataset.data
    return (raw.reshape(len(raw), -1).numpy() / 255.0).astype(np.float32)


#Main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _victim_train, attacker_pool, _validation, test = vt.get_data()

    # attacker: shuffle the pool once, query the largest budget, and use prefixes for the smaller ones
    rng = np.random.default_rng(vt.SEED)
    pool = get_images(attacker_pool)[rng.permutation(len(attacker_pool))][:max(BUDGETS)]
    print(f"querying victim API with {len(pool)} attacker_pool images ({CLIENT_ID})")
    pool_probs = query_victim(pool)

    # evaluator: true test labels + victim's test predictions 
    X_test = get_images(test)
    y_test = test.targets.numpy()
    print(f"querying victim API with {len(X_test)} test images ({EVAL_CLIENT_ID})")
    victim_test_pred = query_victim(X_test, client_id=EVAL_CLIENT_ID).argmax(1)
    victim_acc = float((victim_test_pred == y_test).mean())
    print(f"victim test accuracy (via API): {victim_acc:.4f}")

    rows, best = [], None
    for budget in BUDGETS:
        start = time.perf_counter()
        model = train_substitute(pool[:budget], pool_probs[:budget])
        train_s = time.perf_counter() - start
        sub_pred = predict(model, X_test)
        acc = float((sub_pred == y_test).mean())
        fid = float((sub_pred == victim_test_pred).mean())
        rows.append({"query_budget": budget, "substitute_test_acc": acc, "fidelity": fid,
                     "victim_test_acc": victim_acc, "train_seconds": round(train_s, 1)})
        print(f"budget={budget:5d} | substitute acc={acc:.4f} fidelity={fid:.4f} | "
              f"victim acc={victim_acc:.4f} | train {train_s:.0f}s")
        if best is None or fid > best[0]:
            best = (fid, budget, model)

    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    model_path = os.path.join(OUT_DIR, "substitute_best.pt")
    torch.save(best[2].state_dict(), model_path)
    print(f"best substitute: budget={best[1]} fidelity={best[0]:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(BUDGETS, [r["fidelity"] for r in rows], "o-", color="#c0392b",
            label="Fidelity (agrees with victim)")
    ax.plot(BUDGETS, [r["substitute_test_acc"] for r in rows], "s--", color="#2471a3",
            label="Substitute test accuracy")
    ax.axhline(victim_acc, color="gray", linestyle=":", label=f"Victim test accuracy ({victim_acc:.4f})")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_title("Substitute CNN trained on API soft labels", fontsize=11)
    ax.set_xlabel("Query budget (attacker_pool images sent to /predict)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.8, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.suptitle("VictimNet (MNIST) under model extraction", fontsize=12)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "extraction_mnist.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved {csv_path}, {png_path} and {model_path}")


if __name__ == "__main__":
    main()
