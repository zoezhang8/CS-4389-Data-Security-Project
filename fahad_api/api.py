"""Victim model serving API with query logging (Fahad's part).

Endpoints
    POST /predict   body: {"image": <784 numbers or 28x28 nested list, values 0-1>}
                    -> {"prediction": int, "probabilities": [10 floats], "model": str}
    GET  /health    -> which model is loaded, how many requests were logged

Every request to /predict (valid or not) is appended as one JSON line to
logs/queries.jsonl so the detection monitor can read it.

Run:
    python api.py                      # http://127.0.0.1:5000
    MODEL=dummy python api.py          # force the dummy model
    MODEL=victim python api.py         # require Hafsah's model, fail if missing
Environment variables
    MODEL        auto (default) | victim | dummy
    VICTIM_DIR   folder holding victim_train.py / victim.pt
                 (default: ../ai_security_project_victim)
    LOG_PATH     default logs/queries.jsonl
    RETURN_PROBS 1 (default) returns full probabilities, 0 returns label only
"""
import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.environ.get("LOG_PATH", os.path.join(HERE, "logs", "queries.jsonl"))
VICTIM_DIR = os.environ.get("VICTIM_DIR", os.path.join(os.path.dirname(HERE), "ai_security_project_victim"))
RETURN_PROBS = os.environ.get("RETURN_PROBS", "1") == "1"

_log_lock = threading.Lock()


# --------------------------------------------------------------------------- models
class DummyModel:
    """Stand-in until Hafsah's model is available. Fixed random linear classifier.

    Same interface as the real one: predict_proba(x[784]) -> 10 probabilities.
    Its outputs are NOT meaningful digit predictions.
    """
    name = "dummy"

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.05, size=(784, 10))

    def predict_proba(self, x):
        z = x @ self.w
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()


class VictimModel:
    """Wraps Hafsah's saved PyTorch model (victim.pt)."""
    name = "victim"

    def __init__(self, victim_dir):
        import torch  # imported here so the dummy path works without torch
        sys.path.insert(0, victim_dir)
        from victim_train import load_victim  # Hafsah's loader
        self.torch = torch
        self.model = load_victim(os.path.join(victim_dir, "victim.pt"))

    def predict_proba(self, x):
        t = self.torch.tensor(x, dtype=self.torch.float32).reshape(1, 1, 28, 28)
        with self.torch.no_grad():
            return self.torch.softmax(self.model(t), dim=1)[0].numpy()


def load_model():
    mode = os.environ.get("MODEL", "auto").lower()
    if mode == "dummy":
        return DummyModel()
    try:
        return VictimModel(VICTIM_DIR)
    except Exception as e:  # noqa: BLE001
        if mode == "victim":
            raise
        print(f"WARNING: could not load victim model ({e!r}). Using DUMMY model; "
              f"results are not real. Set MODEL=victim to make this an error.", file=sys.stderr)
        return DummyModel()


MODEL = load_model()
print(f"Serving model: {MODEL.name}", file=sys.stderr)


# --------------------------------------------------------------------------- logging
def write_log(entry):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    line = json.dumps(entry, separators=(",", ":"))
    with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_image(payload):
    """Return a float64 vector of 784 values in [0, 1], or raise ValueError."""
    if not isinstance(payload, dict) or "image" not in payload:
        raise ValueError('body must be JSON like {"image": [...]}')
    try:
        arr = np.asarray(payload["image"], dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("image must contain only numbers")
    if arr.shape not in ((784,), (28, 28)):
        raise ValueError(f"image must have 784 values or be 28x28, got shape {arr.shape}")
    arr = arr.reshape(784)
    if not np.all(np.isfinite(arr)):
        raise ValueError("image contains NaN or infinity")
    if arr.min() < 0 or arr.max() > 1:
        raise ValueError("pixel values must be between 0 and 1")
    return arr


# --------------------------------------------------------------------------- app
app = Flask(__name__)


@app.post("/predict")
def predict():
    start = time.perf_counter()
    entry = {
        "request_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "client_ip": request.remote_addr,
        "client_id": request.headers.get("X-Client-Id", ""),
        "model": MODEL.name,
    }
    try:
        payload = request.get_json(silent=True)
        x = parse_image(payload)
    except ValueError as e:
        entry.update(status=400, error=str(e), latency_ms=round((time.perf_counter() - start) * 1000, 2))
        write_log(entry)
        return jsonify(error=str(e)), 400

    probs = MODEL.predict_proba(x)
    pred = int(np.argmax(probs))
    entry.update(
        status=200,
        # hash lets the monitor spot repeated / near-duplicate queries without storing images
        input_hash=hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest()[:16],
        input_mean=round(float(x.mean()), 4),
        input_nonzero=int((x > 0).sum()),
        prediction=pred,
        confidence=round(float(probs[pred]), 4),
        entropy=round(float(-(probs * np.log(probs + 1e-12)).sum()), 4),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
    write_log(entry)

    body = {"prediction": pred, "model": MODEL.name}
    if RETURN_PROBS:
        body["probabilities"] = [round(float(p), 6) for p in probs]
    return jsonify(body)


@app.get("/health")
def health():
    count = 0
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            count = sum(1 for _ in f)
    return jsonify(status="ok", model=MODEL.name, logged_requests=count)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
