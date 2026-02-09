#!/usr/bin/env bash
# Run eval for three checkpoints (alpha 0.3, 0.5, 0.7)
# Activate your conda env first if needed: conda activate <your_env>

TEST_DIR="data/seval_qwen3_8b_dataset/test/"

for CKPT_PATH in \
  "ckpt/model_epoch_19_alpha_eq_0.pt" \
  "ckpt/model_epoch_19_alpha_eq_0.5.pt" \
  "ckpt/model_epoch_19_alpha_eq_1.0.pt"; do
  echo "========== Evaluating $CKPT_PATH =========="
  python eval.py \
    --ckpt_path "$CKPT_PATH" \
    --test_dataset_dir "$TEST_DIR" \
    --model_name "Qwen/Qwen3-8B" \
    --idx_layer 32
done
