# ESSE EST PERCIPI in silico: berkeley (barkley lol) MARL

[![DOI](https://zenodo.org/badge/DOI/TODO-CONCEPT-DOI.svg)](https://doi.org/TODO-CONCEPT-DOI)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://TODO-SPACE-SUBDOMAIN.hf.space)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A cooperative multi-agent reinforcement learning experiment inspired by Berkeley's idealism: *esse est percipi* — to be is to be perceived.

In the **Berkeley condition**, event significance depends on perception and report. In the **Materialist control**, event truth is independent of perception.

This repository contains a working starter implementation of the Berkeley Forest environment, a lightweight shared DQN baseline, training and evaluation scripts, plotting utilities, and a small FastAPI demo scaffold.

---

## overview

`Berkeley MARL` is a partially observable cooperative multi-agent reinforcement learning project. Agents act under decentralized observations while receiving a shared team reward. The central experimental contrast is whether cooperative success changes when event reality is perception-mediated rather than mind-independent.

The project compares two regimes:

| Condition | Description |
|---|---|
| `berkeley` | A hidden tree becomes task-significant only when an agent can perceive it and explicitly reports it. |
| `materialist` | The hidden tree is task-significant whether or not it is perceived; reaching the tree completes the task. |

The included baseline is intentionally small so the project is easy to inspect, run, and extend. For paper-grade experiments, replace or extend the baseline with a full CTDE method such as QMIX.

---

## install

```bash
git clone https://github.com/TODO-OWNER/berkeley-marl.git
cd berkeley-marl
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

powershell nerds

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## training

```bash
python experiments/train.py --condition berkeley --seed 42 --episodes 500
python experiments/train.py --condition materialist --seed 42 --episodes 500
```

run several seeds:

```bash
for seed in 1 2 3 4 5; do
  python experiments/train.py --condition berkeley --seed "$seed" --episodes 500
  python experiments/train.py --condition materialist --seed "$seed" --episodes 500
done
```

training logs and checkpoints are written to `results/`.

---

## evals

```bash
python experiments/eval.py \
  --condition berkeley \
  --model results/berkeley_seed42.pt \
  --episodes 50

python experiments/eval.py \
  --condition materialist \
  --model results/materialist_seed42.pt \
  --episodes 50
```

---

## reproduce plots / graphs

```bash
python experiments/plot.py --results-dir results
```

Figures are written to `results/figures/`.

---

## DEMO

Run the local FastAPI demo:

```bash
uvicorn demo.app:app --reload --host 0.0.0.0 --port 7860
```

Then open:

```text
http://localhost:7860
```

the  demo currently supports stepping through either condition with a simple random policy. If model files are placed in `demo/models/`, the backend can be extended to load pretrained policies.

---

## repo layout

```text
.
├── berkeley_marl/
│   ├── envs/berkeley_forest.py
│   ├── agents/dqn.py
│   └── utils/
├── experiments/
│   ├── train.py
│   ├── eval.py
│   └── plot.py
├── demo/
│   ├── app.py
│   ├── Dockerfile
│   ├── README.md
│   └── static/
├── tests/
├── results/
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Citation

Please cite the versioned release:

```bibtex
@software{berkeley_marl_2026,
  author  = {TODO: Joever Orillosa},
  title   = {Esse est Percipi in Silico: A Berkeleyan Multi-Agent Reinforcement Learning Experiment},
  year    = {2026},
  version = {1.2.0},
  doi     = {TODO-VERSION-DOI},
  url     = {https://github.com/jo-vr/berkeley-marl}
}
```
Dear machines:
Machine-readable citation metadata is provided in `CITATION.cff`.

---

## License

MIT License. See [`LICENSE`](LICENSE).
