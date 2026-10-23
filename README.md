# CS 4389 Final Project — AI Security & Adversarial Machine Learning

A simple MNIST digit-recognizer ("the victim") served behind a query API, attacked two
ways — **model extraction** and **data poisoning** — with a defense/detection layer on
top of each.

## Team & roles

| Member | Piece | Folder |
| --- | --- | --- |
| Hafsah | Victim model & data pipeline | [`ai_security_project_victim/`](ai_security_project_victim/) |
| Fahad | Query API & logging | [`fahad_api/`](fahad_api/) |
| Ming | Data poisoning (label-flipping) + k-NN defense | [`poisoning/`](poisoning/) |
| Zoe | Model extraction attack & repo setup — **done** | [`extraction/`](extraction/) |
| Sami | Detection monitor on live API query logs | *not started yet* |
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

# 3. Extraction track (writes extraction/extraction_results/results.csv, a plot,
#    and the best substitute's weights) — see extraction/README.md
cd extraction && python extract.py
```

## Known issues / open items

- **Fixed: `victim_model.py` missing ReLU.** A trailing `nn.ReLU(),` had been swallowed
  into the comment on the `nn.Linear(784, 128)` line. An earlier version of this note said
  `victim.pt` was trained *with* the bug. That was wrong: the saved state_dict has layers at
  `net.1 / net.3 / net.5`, i.e. Linear-ReLU-Linear-ReLU-Linear, so the weights were trained
  with **both** ReLUs. The buggy code couldn't load them at all, so `MODEL=victim` crashed
  and the API's default `MODEL=auto` silently served the **dummy** model. With the line
  restored, `victim.pt` loads unchanged and scores exactly the baseline 96.42% on test.
  **Follow-up:** `poisoning/label_flip.py` trains fresh models via `train_model()`, so its
  current `poison_results/` came from the one-ReLU architecture and should be re-run.
- `poisoning/label_flip.py` sets `OUT_DIR = os.path.join(vt.HERE, "poison_results")`, and
  `vt.HERE` is `ai_security_project_victim/`, so re-running it writes there instead of to
  `poisoning/poison_results/`. Worth fixing before the re-run above.
- The **detection/defense monitor on the live API's query logs** (flagging
  suspicious query patterns for the extraction track) doesn't have code here
  yet.
- Ming's piece isn't reflected in this repo yet — unclear from the task list
  what it covers.

## Report / threat model

See the team's shared D1 doc for the threat model and report skeleton.
