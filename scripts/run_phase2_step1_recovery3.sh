#!/bin/bash
# Phase 2 Step 1 recovery v3 — eval R2-v2 + R3-v2 BEFORE R4-v2 so the
# partial R1-R3 table is available as soon as R3-v2 finishes training.
# Then train R4-v2 + eval. Triggers two milestone markers in the log:
#   "PARTIAL_TABLE_READY"  -> R1/R2/R3 rows complete (R4 still pending)
#   "PIPELINE_DONE"        -> all four rows complete
set -o pipefail
cd /workspace/DisMo
PORT=29580
EVAL="python scripts/eval_checkpoint.py --dataset imagenet100 --data-root data --num-workers 14 --bn-recalib 1"
step(){ echo "######## $1 $(date -u +%Y-%m-%dT%H:%M:%S) ########"; }
TR(){ CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$PORT \
      scripts/train.py --config "$1" 2>&1; }

step "TRAIN R3-v2 (simsiam, eff lr 0.05, 80ep)"
TR configs/experiment/phase2/r3_simsiam_v2.yaml
rc=$?; echo "R3-v2 train rc=$rc"
[ $rc -ne 0 ] && { echo "ABORT_PIPELINE R3-v2 rc=$rc"; exit 1; }

step "EVAL R2-v2 + R3-v2 (recalib, parallel)"
CUDA_VISIBLE_DEVICES=0 $EVAL --run-name r2_supervised_v2 \
  --checkpoint checkpoints/phase2/r2_supervised_v2/final.pt \
  --output results/phase2/eval_r2_supervised_v2.json > logs/eval_r2v2.log 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $EVAL --run-name r3_simsiam_v2 \
  --checkpoint checkpoints/phase2/r3_simsiam_v2/final.pt \
  --output results/phase2/eval_r3_simsiam_v2.json > logs/eval_r3v2.log 2>&1 &
B=$!
wait $A; rA=$?; wait $B; rB=$?
echo "R2-v2 eval rc=$rA  R3-v2 eval rc=$rB"
[ $rA -ne 0 ] || [ $rB -ne 0 ] && { echo "ABORT_PIPELINE eval R2/R3 rA=$rA rB=$rB"; exit 1; }

# Render partial table (R4 still pending) — this is the user's first milestone.
python scripts/make_phase2_step1_table.py \
  --out results/phase2/step1_table_partial.md 2>&1
echo "PARTIAL_TABLE_READY $(date -u +%Y-%m-%dT%H:%M:%S)"

step "TRAIN R4-v2 (distill mild aug, 80ep)"
TR configs/experiment/phase2/r4_distill_l2_v2.yaml
rc=$?; echo "R4-v2 train rc=$rc"
[ $rc -ne 0 ] && { echo "ABORT_PIPELINE R4-v2 rc=$rc"; exit 1; }

step "EVAL R4-v2 (recalib)"
CUDA_VISIBLE_DEVICES=0 $EVAL --run-name r4_distill_l2_v2 \
  --checkpoint checkpoints/phase2/r4_distill_l2_v2/final.pt \
  --output results/phase2/eval_r4_distill_l2_v2.json > logs/eval_r4v2.log 2>&1
rc=$?; echo "R4-v2 eval rc=$rc"
[ $rc -ne 0 ] && { echo "ABORT_PIPELINE eval R4-v2 rc=$rc"; exit 1; }

# Final complete table.
python scripts/make_phase2_step1_table.py \
  --out results/phase2/step1_table_final.md 2>&1
echo "PIPELINE_DONE $(date -u +%Y-%m-%dT%H:%M:%S)"
