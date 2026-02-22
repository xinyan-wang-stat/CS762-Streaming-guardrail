# Streaming Guardrail

This is the official repo for our course project of CS762 in fall 2026. 

**Setup:** Create a conda env and install dependencies:
```bash
conda create -n stream python=3.10 -y
conda activate stream
pip install -r requirements.txt
```

**Folders:**
- **data/** — Sample/annotated datasets and annotation script for building safety labels.
- **scripts/** — Shell scripts to run training and evaluation (e.g. different α).
- **utils/** — Download benchmark data (e.g. StreamGuardBench) and convert CSV to HuggingFace dataset.
- **visualization/** — Plot training curves, F1, heatmaps, harmful-token stats and parse result logs.

**Core files:**
- **dataset.py** — Dataset that loads safety data and builds per-sample cached hidden states + token/response labels for training.
- **models.py** — Streaming safety head (attention + CfcCell) that predicts token-level and response-level safety from cached hidden states.
- **train.py** — Trains the safety head on cached data and runs evaluation on the test set.
- **eval.py** — Loads a checkpoint and evaluates the safety head (response-level, streaming-level, and offset metrics; can save logits to JSONL).
