# Model extraction attack

Steals a working copy of the victim model using only its query API (`fahad_api/`).

## Setup
    pip install -r ../fahad_api/requirements.txt matplotlib
Folder layout: this folder sits next to `ai_security_project_victim/` and `fahad_api/`.
No server is needed; the script calls the API in-process through `api.app.test_client()`
with `MODEL=victim` (fails instead of silently using the dummy model).

## Run
    python extract.py
Takes a few minutes on CPU (30,000 API calls + 5 substitute trainings).
Outputs go to `extraction_results/`:
`results.csv`, `extraction_mnist.png` (budget vs. accuracy/fidelity), and
`substitute_best.pt` (state_dict of the `SubstituteNet` with the highest fidelity).

## Threat model
- Attacker can only `POST /predict` and read back `prediction` + `probabilities`.
- No access to `victim_train`, `test`, the victim's weights, or its code/architecture.
- Query inputs: `attacker_pool` **images** (unlabeled, in-distribution digits). Its true
  labels are never read.
- Substitute is a small CNN, deliberately not VictimNet's MLP, since a black-box attacker
  doesn't know the victim's architecture (see the `SubstituteNet` docstring).
- Trained with soft-label cross-entropy against the returned probability vectors.
  Budgets are nested prefixes of one seeded shuffle of `attacker_pool`.

## Results columns (`results.csv`, one row per query budget)
    query_budget         number of attacker_pool images sent to /predict and used for training
    substitute_test_acc  substitute's accuracy on the 10k test set vs. true labels
    fidelity             fraction of test images where substitute and victim predict the same
                         label (the main extraction metric)
    victim_test_acc      victim's own test accuracy (via the API), for reference
    train_seconds        wall-clock time to train that substitute

Current run: fidelity 0.944 at 1,000 queries, 0.970 at 2,500, 0.978 at 20,000.
Substitute accuracy passes the victim's (0.9642) from about 5,000 queries on.

## Notes for teammates
- Log entries: attack queries use `X-Client-Id: zoe-extraction`. Scoring queries
  (the victim's predictions on `test`, needed to compute fidelity) use
  `zoe-extraction-eval`. The detection monitor should treat only the first as attack
  traffic. Logs land in `fahad_api/logs/queries.jsonl`.
- Defense to compare: `RETURN_PROBS=0` (labels only) removes the soft labels this
  attack trains on.
