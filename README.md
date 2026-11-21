# CS 4389 Final Project — AI Security & Adversarial Machine Learning

This project examines attacks against an MNIST handwritten-digit classifier and evaluates detection and defense techniques. A trained PyTorch **victim model** is exposed through a Flask prediction API. The project demonstrates **model extraction** and **training-data poisoning**, then compares their outcomes with and without defenses.

## Features

- **Victim model:** A fully connected MNIST classifier with saved baseline weights.
- **Prediction API:** Flask endpoints for predictions and health checks, with JSONL query logging.
- **Model extraction:** Trains a substitute CNN using probability vectors returned by the API.
- **Extraction defense:** Configurable per-client query quota and optional time-based rate limit. Compares unrestricted and defended accuracy/fidelity.
- **Detection monitor:** Standalone log analyzer generating alerts for rapid querying, high cumulative query counts, and repeated HTTP 429 responses. Supports replay and live monitoring.
- **Data poisoning:** Random and targeted (digit 7 → digit 1) label-flipping experiments, comparing no defense, k-nearest-neighbor (k-NN) filtering, and generalized cross entropy (GCE).

## System Requirements

| Item | Requirement |
| --- | --- |
| Operating system | Windows 10/11 recommended. Linux/macOS should also work with equivalent commands and compatible packages. |
| Programming language | Python 3; Python 3.11 recommended. |
| Software | Python, pip, optionally Git and VS Code. |
| Dependencies | Flask, NumPy, PyTorch, Torchvision, Matplotlib, scikit-learn. |
| Hardware | CPU supported; GPU not required. |
| Internet | Required to download MNIST data on the first run if not already cached. |

No database, Docker container, paid service, or separate frontend is required.

## Installation Instructions

Run all commands from the repository root in a Windows PowerShell terminal.

### 1. Clone the repository

```powershell
git clone https://github.com/zoezhang8/CS-4389-Data-Security-Project.git
cd CS-4389-Data-Security-Project
```

Alternatively, download the repository ZIP through GitHub and extract it.

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\fahad_api\requirements.txt
```

### 4. Verify installation

```powershell
python -c "import flask, numpy, torch, torchvision, matplotlib, sklearn; print('Dependencies OK')"
```

For Linux/macOS, activate the environment with `source .venv/bin/activate` and use the appropriate shell path separators.

On the first dataset-dependent run, MNIST is downloaded automatically into the project's `data/` folder if needed.

## Running the Project

### 1. Victim Model

The trained victim model is provided in `ai_security_project_victim/victim.pt`. Retraining is not required.

The fixed-seed MNIST dataset contains:

| Split | Images | Purpose |
| --- | ---: | --- |
| Victim training | 30,000 | Train victim; simulate compromised labels during poisoning |
| Attacker pool | 20,000 | Model extraction inputs |
| Validation | 10,000 | Reserved validation data |
| Test | 10,000 | Accuracy and fidelity evaluation |

The saved victim model has a baseline test accuracy of **96.42%**.

To retrain the model intentionally:

```powershell
python .\ai_security_project_victim\victim_train.py
```

**Note:** Retraining overwrites the saved victim weights and baseline metadata.

### 2. Prediction API

Start the real victim API:

```powershell
$env:MODEL = "victim"
$env:RETURN_PROBS = "1"
$env:RATE_LIMIT_ENABLED = "0"

python .\fahad_api\api.py
```

The API runs at `http://127.0.0.1:5000`.

Open a second terminal and verify that it is running:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Send a test prediction:

```powershell
$pixels = @(0.0) * 784
$body = @{ image = $pixels } | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri http://127.0.0.1:5000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Headers @{"X-Client-Id"="manual-test"} `
  -Body $body
```

The `/predict` endpoint accepts 784 normalized pixel values or a 28×28 array, with values between 0 and 1.

A successful response includes a digit prediction, the model name, and optionally a probability vector. Invalid inputs return HTTP 400.

Requests are recorded in:

`fahad_api/logs/queries.jsonl`

Stop the server using `Ctrl+C`.

### 3. API Rate-Limiting Defense

Rate limiting is disabled by default.

To enable a maximum of 1,000 accepted requests per client, stop the API and restart it with:

```powershell
$env:MODEL = "victim"
$env:RATE_LIMIT_ENABLED = "1"
$env:RATE_LIMIT_MAX_QUERIES = "1000"
$env:RATE_LIMIT_PER_WINDOW = "0"

python .\fahad_api\api.py
```

After the first 1,000 allowed queries, additional requests return HTTP 429.

To use a time-based limit instead:

```powershell
$env:RATE_LIMIT_ENABLED = "1"
$env:RATE_LIMIT_MAX_QUERIES = "0"
$env:RATE_LIMIT_PER_WINDOW = "100"
$env:RATE_LIMIT_WINDOW_SECONDS = "60"

python .\fahad_api\api.py
```

This permits up to 100 requests per client per 60-second window.

Restart the API when changing environment settings.

### 4. Model Extraction Attack and Defense Comparison

The extraction script communicates with the Flask API through an in-process test client. **The standalone API server does not need to be running.**

Execute:

```powershell
$env:RETURN_PROBS = "1"
python .\extraction\extract.py --mode both --quota 1000
```

The script evaluates the victim, then performs two experiments.

**Unrestricted extraction:** The attacker can obtain all 20,000 prediction responses.

**Defended extraction:** The attacker attempts 20,000 queries but is restricted to 1,000 successful responses. The remaining requests return HTTP 429.

Substitute models are trained at attempted query budgets of 1,000, 2,500, 5,000, 10,000, and 20,000.

The experiment measures:

- **Test accuracy:** Agreement with true MNIST labels.
- **Fidelity:** Agreement between the substitute and victim predictions.
- **Successful queries:** Prediction responses received.
- **Blocked queries:** Requests rejected by the defense.
- **Training time:** Time spent training each substitute model.

Additional supported commands:

```powershell
python .\extraction\extract.py --mode unrestricted
python .\extraction\extract.py --mode defended --quota 1000
python .\extraction\extract.py --mode both --quota 2500
```

Results are saved in `extraction/extraction_results/`:

| Output file | Description |
| --- | --- |
| `results_unrestricted.csv` | Extraction results without query restrictions |
| `results_defended.csv` | Results with query restrictions |
| `extraction_comparison.png` | Accuracy and fidelity comparison graph |
| `substitute_best_unrestricted.pt` | Best unrestricted substitute weights |
| `substitute_best_defended.pt` | Best defended substitute weights |

Running `--mode both` produces the complete comparison graph. Single-mode runs replace that graph with only the selected scenario.

At the 20,000-query budget, the committed results show:

| Metric | Unrestricted | Defended |
| --- | ---: | ---: |
| Substitute test accuracy | 96.93% | 93.04% |
| Fidelity | 97.82% | 94.32% |
| Successful attacker queries | 20,000 | 1,000 |
| Blocked attacker queries | 0 | 19,000 |

Results may vary slightly across runs and hardware.

### 5. Detection Monitor and Alerts

The detection monitor independently examines API request logs for suspicious behavior. It is passive and does not automatically activate a defense.

To analyze previously recorded logs:

```powershell
python .\detection\monitor.py
```

To monitor requests live, open two terminals.

**Terminal 1 — Detection monitor**

```powershell
python .\detection\monitor.py --follow 
```

**Terminal 2 — Extraction experiment**

```powershell
$env:RETURN_PROBS = "1"
python .\extraction\extract.py --mode both --quota 1000
```

The monitor generates terminal alerts and saves JSONL alert records.

| Detection rule | Trigger |
| --- | --- |
| `HIGH_QUERY_RATE` | 500 requests within 60 seconds |
| `HIGH_TOTAL_QUERIES` | 2,000 cumulative requests by a client |
| `REPEATED_RATE_LIMIT_BLOCKS` | 20 HTTP 429 responses within 60 seconds |

Alerts include timestamp, client ID, severity, detection rule, and count.

Replaying the same query logs multiple times will generate duplicate alerts in alerts.jsonl, because the monitor only prevents duplicate alerts within a single execution.

### 6. Data Poisoning Attack and Defenses

Run:

```powershell
python .\poisoning\label_flip.py
```

The experiment tests two label-flipping attacks.

**Random label flipping:** Training labels are randomly changed to incorrect classes.

**Targeted label flipping:** Selected training examples of digit 7 are relabeled as digit 1.

Poisoning rates of 0%, 5%, 10%, 20%, 30%, and 40% are tested.

Each scenario compares three approaches:

1. **No defense:** Ordinary training on poisoned labels.
2. **k-NN filtering:** Suspicious samples are identified through neighboring-label consistency and removed before retraining.
3. **Generalized Cross Entropy (GCE):** A loss function that reduces the influence of inconsistent labels during training (`q = 0.7`).

Results are saved in `poisoning/poison_results/`:

- `results.csv` — Test accuracy, digit-7 recall, k-NN detection rates, removed-sample counts, and GCE performance.
- `label_flip_mnist.png` — Two-panel graph comparing all three methods.

The left graph compares test accuracy under random label flipping. The right graph compares digit-7 recall under targeted label flipping.

The experiment involves repeated training and k-NN analysis and may require significant CPU processing time.

Neither defense is guaranteed to improve every metric at every poisoning rate.

## Testing Instructions

The repository includes automated API tests.

Run:

```powershell
$env:RATE_LIMIT_ENABLED = "0"
$env:RETURN_PROBS = "1"

python .\fahad_api\test_api.py
```

These tests verify:

- Valid flattened and 2D image inputs.
- Rejection and logging of invalid inputs.
- Correct request-log fields.
- Consistent hashing of identical inputs.
- API health endpoint behavior.

The automated tests use a dummy model and temporary logs. They do not alter the saved trained victim model.

Rate limiting, extraction defenses, poisoning defenses, and detection alerts can be verified using the commands in the preceding sections.

## Login Credentials for Testing

**No login credentials are required.**

The application does not implement a user login system or API authentication. No username, password, or API token is necessary to execute the experiments locally.

The optional `X-Client-Id` header identifies requests for logging and rate-limit grouping. It is not an authenticated credential.

## Limitations and Incomplete Features

**1. Rate-limiting limitations**

Rate-limit counters are stored in memory and reset whenever the API restarts. Different API processes do not share counters. Because client identifiers are not authenticated, attackers could potentially evade limits by changing identifiers.

The extraction experiment uses a cumulative query quota for reproducible results. This differs from a conventional time-based rate limit, which an attacker may circumvent by delaying requests.

**2. Detection monitor limitations**

The monitor is passive. Alerts are displayed in the terminal and written to a file, but they do not automatically block clients, change rate limits, or send email notifications.

Detection thresholds are configured for demonstration purposes and may produce false positives or miss slower attacks. The monitor does not include advanced log rotation or persistent alert-state management.

**3. Probability disclosure**

The API returns probability vectors when `RETURN_PROBS=1`, making model extraction easier. Setting `RETURN_PROBS=0` hides probabilities, but the current extraction training procedure requires probability vectors and cannot operate in label-only mode.

**4. Victim model fallback**

The standalone API's default `MODEL=auto` may load a dummy model if the trained victim cannot be loaded. Use `MODEL=victim` when conducting actual experiments.

**5. Dataset and computational limitations**

The experiments use MNIST and relatively small neural networks. Their conclusions may not generalize directly to larger or more complex AI systems.

Full poisoning and extraction experiments may be time-consuming on CPU-only systems. Training results and execution times may vary depending on hardware and installed package versions.

## Team & roles

| Member | Piece | Folder |
| --- | --- | --- |
| Hafsah | Victim model & data pipeline | [`ai_security_project_victim/`](ai_security_project_victim/) |
| Fahad | Query API & logging | [`fahad_api/`](fahad_api/) |
| Ming | Data poisoning (label-flipping) + k-NN defense | [`poisoning/`](poisoning/) |
| Zoe | Model extraction attack & repo setup — **done** | [`extraction/`](extraction/) |
| Sami | Detection monitor on live API query logs & Secondary Defenses | [`detection/`](detection/) |
| Albion | Threat model, D1/D2 report | *see team doc* |
