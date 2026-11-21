""" Zoe Zhang, 2024-06-05
Model extraction track: steal a copy of the victim model through its query API

Attack:
  1. query_victim() sends attacker_pool images to /predict and keeps the returned probability vectors as soft labels.
  2. A substitute CNN is trained on pairs for several query budgets. Budgets are nested
     prefixes of one shuffled order, so budget 20000 = the whole pool

run python extract.py
outputs go to ./extraction_results/ (results.csv, extraction_mnist.png,substitute_best.pt)
"""
import argparse
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
def query_victim(images, client_id):
    """Query API and collect only successful responses.

    Returns:
        accepted_indices: positions of successful images
        accepted_probs: corresponding probability vectors
    """

    client = api.app.test_client()

    accepted_indices = []
    accepted_probs = []
    blocked = 0

    for i, x in enumerate(images):
        response = client.post(
            "/predict",
            json={"image": x.tolist()},
            headers={"X-Client-Id": client_id},
        )

        if response.status_code == 200:
            data = response.get_json()

            if "probabilities" not in data:
                raise RuntimeError(
                    "API must have RETURN_PROBS=1 for extraction"
                )

            accepted_indices.append(i)
            accepted_probs.append(data["probabilities"])

        elif response.status_code == 429:
            # Defense rejected this request.
            blocked += 1

        else:
            raise RuntimeError(
                f"Query {i} failed: "
                f"{response.status_code} "
                f"{response.get_json()}"
            )

        if (i + 1) % 5000 == 0:
            print(
                f"  {client_id}: "
                f"attempted={i + 1}, "
                f"allowed={len(accepted_indices)}, "
                f"blocked={blocked}"
            )

    return (
        np.asarray(accepted_indices, dtype=np.int32),
        np.asarray(
            accepted_probs, dtype=np.float32
        ).reshape(-1, 10),
    )

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

# -------------------------------------------------- experiments

def run_scenario(
    mode, pool, X_test, y_test,
    victim_test_pred, victim_acc
):
    # Each scenario starts with independent API counters.
    api.reset_rate_limiter()

    # Turn defense on or off for this experiment.
    api.RATE_LIMIT_ENABLED = (mode == "defended")

    client_id = f"{CLIENT_ID}-{mode}"

    print(f"\n== {mode.upper()} ==")
    print(f"Querying {len(pool)} attacker images")

    accepted_indices, accepted_probs = query_victim(
        pool, client_id
    )

    print(
        f"Allowed: {len(accepted_indices)} | "
        f"Blocked: {len(pool) - len(accepted_indices)}"
    )

    rows = []
    best = None

    for budget in BUDGETS:
        # Budget represents attempted queries.
        # Count the successful responses within that budget.
        count = int(
            np.searchsorted(
                accepted_indices, budget, side="left"
            )
        )

        if count == 0:
            raise RuntimeError(
                "No successful queries. Increase the quota."
            )

        # Train only on successfully queried images.
        train_images = pool[accepted_indices[:count]]
        train_probs = accepted_probs[:count]

        start = time.perf_counter()

        model = train_substitute(
            train_images, train_probs
        )

        train_s = time.perf_counter() - start

        # Evaluate substitute against real test labels.
        sub_pred = predict(model, X_test)

        acc = float((sub_pred == y_test).mean())

        # Fidelity = agreement with the real victim.
        fid = float(
            (sub_pred == victim_test_pred).mean()
        )

        row = {
            "query_budget": budget,
            "successful_queries": count,
            "blocked_queries": budget - count,
            "substitute_test_acc": acc,
            "fidelity": fid,
            "victim_test_acc": victim_acc,
            "train_seconds": round(train_s, 1),
        }

        rows.append(row)

        print(
            f"attempts={budget:5d} | "
            f"accepted={count:5d} | "
            f"blocked={budget-count:5d} | "
            f"accuracy={acc:.4f} | "
            f"fidelity={fid:.4f}"
        )

        if best is None or fid > best[0]:
            best = (fid, budget, model)

    # Write separate results for each scenario.
    csv_path = os.path.join(
        OUT_DIR, f"results_{mode}.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    # Preserve the best substitute for this scenario.
    model_path = os.path.join(
        OUT_DIR, f"substitute_best_{mode}.pt"
    )

    torch.save(best[2].state_dict(), model_path)

    print(f"Saved {csv_path}")
    print(f"Saved {model_path}")

    return rows


# -------------------------------------------------- graph

def plot_comparison(results, victim_acc):
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        "unrestricted": ("o-", "s--"),
        "defended": ("^-", "d--"),
    }

    for mode, rows in results.items():
        x = [r["query_budget"] for r in rows]

        accuracy = [
            r["substitute_test_acc"] for r in rows
        ]

        fidelity = [
            r["fidelity"] for r in rows
        ]

        ax.plot(
            x, accuracy, styles[mode][0],
            label=f"{mode.title()}: accuracy"
        )

        ax.plot(
            x, fidelity, styles[mode][1],
            label=f"{mode.title()}: fidelity"
        )

    ax.axhline(
        victim_acc,
        linestyle=":",
        color="gray",
        label=f"Victim test accuracy ({victim_acc:.4f})"
    )

    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(x) for x in BUDGETS])

    ax.set_xlabel("Attempted API queries")
    ax.set_ylabel("Score")
    ax.set_ylim(0.8, 1.0)

    ax.set_title(
        "Model extraction: API defense comparison"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_path = os.path.join(
        OUT_DIR, "extraction_comparison.png"
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


# -------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=("both", "unrestricted", "defended"),
        default="both",
    )

    parser.add_argument(
        "--quota",
        type=int,
        default=1000,
        help="Maximum allowed queries in defended mode",
    )

    args = parser.parse_args()

    if args.quota < 1:
        parser.error("--quota must be at least 1")

    os.makedirs(OUT_DIR, exist_ok=True)

    if not api.RETURN_PROBS:
        raise RuntimeError(
            "Extraction requires RETURN_PROBS=1"
        )

    # Use a hard query quota for reproducible comparison.
    # Disable the optional time-window limit here.
    api.RATE_LIMIT_MAX_QUERIES = args.quota
    api.RATE_LIMIT_PER_WINDOW = 0
    api.RATE_LIMIT_ENABLED = False
    api.reset_rate_limiter()

    # Load the same seeded MNIST split for both experiments.
    _victim_train, attacker_pool, _validation, test = (
        vt.get_data()
    )

    rng = np.random.default_rng(vt.SEED)

    pool = get_images(attacker_pool)[
        rng.permutation(len(attacker_pool))
    ][:max(BUDGETS)]

    X_test = get_images(test)
    y_test = test.targets.numpy()

    # Trusted evaluation happens before enabling the limit.
    # These labels/predictions are never used for
    # substitute training.
    print(
        f"Evaluating victim with {len(X_test)} test images"
    )

    eval_indices, eval_probs = query_victim(
        X_test, EVAL_CLIENT_ID
    )

    if len(eval_indices) != len(X_test):
        raise RuntimeError(
            "Evaluation should not be rate limited"
        )

    victim_test_pred = eval_probs.argmax(axis=1)

    victim_acc = float(
        (victim_test_pred == y_test).mean()
    )

    print(f"Victim test accuracy: {victim_acc:.4f}")

    # Run either one scenario or both.
    if args.mode == "both":
        modes = ("unrestricted", "defended")
    else:
        modes = (args.mode,)

    results = {}

    for mode in modes:
        results[mode] = run_scenario(
            mode,
            pool,
            X_test,
            y_test,
            victim_test_pred,
            victim_acc,
        )

    plot_comparison(results, victim_acc)


if __name__ == "__main__":
    main()
