#!/usr/bin/env bash
# Run eval with checkpoint model_epoch_19_aplha_eq_0.5.pt

CKPT_PATH="model_epoch_19_aplha_eq_0.5.pt"
TEST_DIR="data/s_eval/qwen3_8b/test/"

python eval.py \
  --ckpt_path "$CKPT_PATH" \
  --test_dataset_dir "$TEST_DIR" \
  --model_name "Qwen/Qwen3-8B" \
  --idx_layer 32
