import json
import os
import sys
 
# Make `from victim.model import ...` work whether this file is run as a
# script from the project root or imported as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
 
from victim_model import VictimNet
 
SEED = 42
SPLIT_SIZES = (30000, 20000, 10000)  # victim train, attacker pool, validation
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
MODEL_PATH = os.path.join(HERE, "victim.pt")
BASELINE_PATH = os.path.join(HERE, "baseline.json")
 
 
def get_data():
    """Return (victim_train, attacker_pool, validation, test) datasets.
 
    The split is seeded, so every teammate gets exactly the same split.
    """
    full_train = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=ToTensor())
    test = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=ToTensor())
    gen = torch.Generator().manual_seed(SEED)
    victim_train, attacker_pool, validation = random_split(full_train, SPLIT_SIZES, generator=gen)
    return victim_train, attacker_pool, validation, test
 
 
def evaluate(model, dataset, batch_size=256):
    """Return accuracy (0-1) of `model` on `dataset`."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
    return correct / len(dataset)
 
 
def train_model(train_dataset, epochs=5, lr=1e-3, batch_size=64, seed=SEED, verbose=True):
    """Train a fresh VictimNet on `train_dataset` and return it.
 
    `train_dataset` can be any dataset yielding (image_tensor, label) pairs, so
    a poisoned dataset (e.g. with flipped labels) works exactly the same way:
        clean_model    = train_model(victim_train)
        poisoned_model = train_model(poisoned_dataset)
    """
    torch.manual_seed(seed)
    model = VictimNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        if verbose:
            print(f"epoch {epoch}/{epochs}  loss {total_loss / len(train_dataset):.4f}")
    return model
 
 
def save_victim(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
 
 
def load_victim(path=MODEL_PATH):
    """Load the saved victim model (used by the API)."""
    model = VictimNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
 
 
def main():
    victim_train, attacker_pool, validation, test = get_data()
    print(f"victim train: {len(victim_train)}  attacker pool: {len(attacker_pool)}  "
          f"validation: {len(validation)}  test: {len(test)}")
 
    model = train_model(victim_train, epochs=5)
    acc = evaluate(model, test)
    print(f"\nBASELINE TEST ACCURACY: {acc * 100:.2f}%")
 
    save_victim(model)
    reloaded = load_victim()
    acc2 = evaluate(reloaded, test)
    print(f"Accuracy after reloading saved model: {acc2 * 100:.2f}%")
 
    with open(BASELINE_PATH, "w") as f:
        json.dump({
            "baseline_test_accuracy": acc,
            "seed": SEED,
            "split_sizes": {"victim_train": SPLIT_SIZES[0], "attacker_pool": SPLIT_SIZES[1],
                            "validation": SPLIT_SIZES[2], "test": len(test)},
            "epochs": 5,
            "architecture": "784-128-64-10 MLP, Adam lr=1e-3, batch 64",
        }, f, indent=2)
    print(f"Saved {MODEL_PATH} and {BASELINE_PATH}")
 
 
if __name__ == "__main__":
    main()
 