---
title: Berkeley MARL Demo
emoji: 🌳
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Berkeley MARL Demo

Interactive demo for the Berkeley Forest environment.

## REPRODUCE a MINMAL RUN :)

```bash
git clone https://github.com/jo-vr/berkeley-marl.git
cd berkeley-marl

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pytest -q
python experiments/train.py --condition berkeley --episodes 10 --seed 42
python experiments/eval.py --condition berkeley --model results/berkeley_seed42.pt --episodes 10