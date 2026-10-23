# CS 4389 Final Project — AI Security & Adversarial Machine Learning

A simple MNIST digit-recognizer ("the victim") served behind a query API, attacked two
ways — **model extraction** and **data poisoning** — with a defense/detection layer on
top of each.

## Team & roles

| Member | Piece | Folder |
| --- | --- | --- |
| Hafsah | Victim model & data pipeline | [`ai_security_project_victim/`](ai_security_project_victim/) |
| Fahad | Query API & logging | [`fahad_api/`](fahad_api/) |
| Sami | Data poisoning (label-flipping) + k-NN defense | [`poisoning/`](poisoning/) |
| Zoe | Model extraction attack & repo setup | [`extraction/`](extraction/) |
| — | Detection monitor on live API query logs | *not started yet* |
| — | Threat model, D1/D2 report | *see team doc* |

## Shared pieces

Everything imports the same victim model so results across tracks are comparable:

```
ai_security_project_victim/
    victim_model.py   # VictimNet: 784 -> 128 -> 64 -> 10 MLP
    victim_train.py   # get_data(), train_model(), evaluate(), save/load_victim()
    victim.pt          # trained weights (baseline test accuracy: 96.42%)
    baseline.json      # training config + the seeded data split
```

`victim_train.get_data()` returns a **seeded** split of MNIST (same for every
teammate): `victim_train` (30k, used only to train the real model), `attacker_pool`
(20k, **reserved for attack tracks** — extraction and poisoning should draw their
queries/training data from here, not from `victim_train` or `test`), `validation`
(10k), and `test` (10k, used only for scoring).

## Running the pieces

```bash
# 1. API (serves the real model; falls back to a dummy model with a warning if
#    victim.pt isn't found)
cd fahad_api && pip install -r requirements.txt && MODEL=victim python api.py

# 2. Poisoning track (writes poisoning/poison_results/results.csv + a plot)
cd poisoning && python label_flip.py

# 3. Extraction track — see extraction/README.md
```

## Known issues / open items

- **`ai_security_project_victim/victim_model.py` has a formatting bug**: on the
  `nn.Linear(784, 128)` line, a trailing `nn.ReLU(),` got swallowed into the
  inline comment, so the trained network only has **one** ReLU (after the second
  Linear) instead of the two described in the docstring. It still trains fine
  (96.42% test accuracy), but it doesn't match the documented architecture.
  **This needs a team decision before fixing** — the saved `victim.pt` weights
  were trained *with* this bug, and ReLU has no parameters, so just adding the
  missing line back in without retraining would silently change the model's
  behavior without changing its saved weights. Flagging here rather than
  silently patching it.
- The **detection/defense monitor on the live API's query logs** (flagging
  suspicious query patterns for the extraction track) doesn't have code here
  yet.
- Ming's piece isn't reflected in this repo yet — unclear from the task list
  what it covers.

## Report / threat model

See the team's shared D1 doc for the threat model and report skeleton.
