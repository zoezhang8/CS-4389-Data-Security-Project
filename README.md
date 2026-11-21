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
| Sami | Detection monitor on live API query logs & Secondary Defenses | *in progress* |
| Albion | Threat model, D1/D2 report | *see team doc* |

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

The [label-flip graph](poisoning/poison_results/label_flip_mnist.png) compares
ordinary training, k-NN filtering, and generalized cross entropy (GCE, `q=0.7`).
Each method sees the same noisy labels for a given flip rate. GCE keeps every
sample and gives less influence to labels the model assigns low probability.
The left panel shows overall test accuracy under random flips. The right panel
shows recall on test images of the digit 7 when a fraction of training 7s are
relabeled as 1s. In the [results CSV](poisoning/poison_results/results.csv),
`_defended` denotes k-NN and `_gce` denotes generalized cross entropy. Detection
rates and removal counts describe the k-NN filter. The GCE loss and choice of
`q` follow [Zhang and Sabuncu (2018)](https://proceedings.neurips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf).

## Known issues / open items

- **Fixed: `victim_model.py` missing ReLU.** A trailing `nn.ReLU(),` had been swallowed
  into the comment on the `nn.Linear(784, 128)` line. An earlier version of this note said
  `victim.pt` was trained *with* the bug. That was wrong: the saved state_dict has layers at
  `net.1 / net.3 / net.5`, i.e. Linear-ReLU-Linear-ReLU-Linear, so the weights were trained
  with **both** ReLUs. The buggy code couldn't load them at all, so `MODEL=victim` crashed
  and the API's default `MODEL=auto` silently served the **dummy** model. With the line
  restored, `victim.pt` loads unchanged and scores exactly the baseline 96.42% on test.
- The **detection/defense monitor on the live API's query logs** (flagging
  suspicious query patterns for the extraction track) doesn't have code here
  yet.

## Report / threat model

See the team's shared D1 doc for the threat model and report skeleton.
