"""Tests for the API. Run: python -m pytest -q   (or: python test_api.py)"""
import json
import os
import tempfile

os.environ["MODEL"] = "dummy"
os.environ["LOG_PATH"] = os.path.join(tempfile.mkdtemp(), "queries.jsonl")

import api  # noqa: E402

client = api.app.test_client()


def read_log():
    with open(api.LOG_PATH) as f:
        return [json.loads(line) for line in f]


def test_valid_flat_and_2d():
    flat = [0.0] * 784
    grid = [[0.5] * 28 for _ in range(28)]
    for img in (flat, grid):
        r = client.post("/predict", json={"image": img})
        assert r.status_code == 200
        body = r.get_json()
        assert 0 <= body["prediction"] <= 9
        assert len(body["probabilities"]) == 10
        assert abs(sum(body["probabilities"]) - 1) < 1e-3


def test_bad_inputs_rejected_and_logged():
    before = len(read_log())
    bad = [
        {"image": [0.0] * 10},          # wrong size
        {"image": [2.0] * 784},         # out of range
        {"image": ["a"] * 784},         # not numbers
        {"nope": 1},                    # missing key
    ]
    for payload in bad:
        assert client.post("/predict", json=payload).status_code == 400
    assert client.post("/predict", data="not json").status_code == 400
    log = read_log()
    assert len(log) == before + 5
    assert all(e["status"] == 400 and "error" in e for e in log[before:])


def test_every_request_logged_with_fields():
    before = len(read_log())
    client.post("/predict", json={"image": [0.1] * 784}, headers={"X-Client-Id": "alice"})
    e = read_log()[-1]
    assert len(read_log()) == before + 1
    for k in ("request_id", "timestamp", "client_ip", "client_id", "status", "prediction",
              "confidence", "entropy", "input_hash", "latency_ms", "model"):
        assert k in e, k
    assert e["client_id"] == "alice" and e["status"] == 200


def test_same_input_same_hash_different_input_different_hash():
    client.post("/predict", json={"image": [0.2] * 784})
    client.post("/predict", json={"image": [0.2] * 784})
    client.post("/predict", json={"image": [0.3] * 784})
    a, b, c = read_log()[-3:]
    assert a["input_hash"] == b["input_hash"] != c["input_hash"]


def test_health():
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["model"] == "dummy" and body["logged_requests"] >= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
