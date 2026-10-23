# Model extraction attack

Steals a working copy of the victim model using its query API (`fahad_api/`).

## Setup
    pip install -r ../fahad_api/requirements.txt matplotlib
Folder layout: this folder stits next to `ai_security_project_victim/` and `fahad_api/`.
No server needed; script calls the API in-process through `api.app.test_client()`
with `MODEL=victim` (fails instead of silently using the dummy model).

## Run
    python extract.py
Takes a few minutes on CPU (30,000 API calls + 5 substitute trainings).
Outputs go to `extraction_results/`:
`results.csv`, `extraction_mnist.png`, and
`substitute_best.pt` .

## Threat model
- Attacker can only `POST /predict` and read back `prediction` + `probabilities`.
- No access to `victim_train`, `test`, the victim's weights, or code/architecture
- Query inputs: `attacker_pool` images
- Substitute is a small CNN since a black-box attacker doesn't know the victim's architecture.
- Trained with cross-entropy against the returned probability vectors,budgets are nested prefixes of one seeded shuffle of `attacker_pool`.

## Results columns (`results.csv`, one row per query budget)
    query_budget         num of attacker_pool images sent to /predict and used for training
    substitute_test_acc  substitute's accuracy on test set vs true labels
    fidelity             fraction of test images where substitute and victim predict the same label 
    victim_test_acc      victim's test accuracy (via API), testing
    train_seconds        the time to train that substitute

Current run: fidelity 0.944 at 1,000 queries, 0.970 at 2,500, 0.978 at 20,000
Substitute accuracy passes the victim's (0.9642) from about 5,000 queries on


