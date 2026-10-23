# Victim model API with query logging

Flask service that serves the digit recognizer and logs every request.

## Setup
    pip install -r requirements.txt
Folder layout: put this folder next to `ai_security_project_victim/` (or set `VICTIM_DIR`).

## Run
    python api.py                 # auto: real model if it loads, otherwise DUMMY (warning printed)
    MODEL=victim python api.py    # real model or fail
    MODEL=dummy  python api.py    # dummy model
Check `GET /health` to see which model is serving. Every response and log line also carries `model`.

## Use
    curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
         -d '{"image": [0.0, ... 784 values in 0-1 ...]}'
Optional header `X-Client-Id: name` is recorded so the monitor can tell clients apart.

## Log format (`logs/queries.jsonl`, one JSON object per line)
request_id, timestamp (UTC), client_ip, client_id, model, status (200/400), error (400 only),
input_hash, input_mean, input_nonzero, prediction, confidence, entropy, latency_ms.
Bad requests are logged too, since attackers often probe with malformed input.

## Notes for teammates
- `RETURN_PROBS=1` (default) returns full probabilities, which makes extraction easier.
  Set `RETURN_PROBS=0` to return labels only (a possible defense to compare).
- No authentication or rate limiting yet; that is part of the defense work.

## Tests
    python test_api.py
