#!/bin/bash
# Phase 2 Step 1 FINAL recovery pipeline (autonomous, 2 GPUs, 80 ep).
# Fixes: (1) BN recalib at eval (uniform), (2) eff lr 0.05 no batch-scaling
# for SGD R2/R3 (MobileNetV2 LR-sensitive), (3) milder aug for R4.
# R1 (random) already has a BN-recalib eval -> reuse its JSON.
set -o pipefail
cd /workspace/DisMo
PORT=29560
EVAL="python scripts/eval_checkpoint.py --dataset imagenet100 --data-root data --num-workers 14 --bn-recalib 1"
step(){ echo "######## $1 $(date -u +%Y-%m-%dT%H:%M:%S) ########"; }
TR(){ CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$PORT \
      scripts/train.py --config "$1" 2>&1; }

step "TRAIN R2-v2 (supervised, eff lr 0.05, 80ep)"
TR configs/experiment/phase2/r2_supervised_v2.yaml
rc=$?; echo "R2-v2 train rc=$rc"; [ $rc -ne 0 ] && { echo "ABORT_PIPELINE R2-v2"; exit 1; }

step "TRAIN R3-v2 (simsiam, eff lr 0.05, 80ep)"
TR configs/experiment/phase2/r3_simsiam_v2.yaml
rc=$?; echo "R3-v2 train rc=$rc"; [ $rc -ne 0 ] && { echo "ABORT_PIPELINE R3-v2"; exit 1; }

step "TRAIN R4-v2 (distill mild aug, 80ep)"
TR configs/experiment/phase2/r4_distill_l2_v2.yaml
rc=$?; echo "R4-v2 train rc=$rc"; [ $rc -ne 0 ] && { echo "ABORT_PIPELINE R4-v2"; exit 1; }

step "EVAL R2-v2 + R3-v2 (recalib)"
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
[ $rA -ne 0 ] || [ $rB -ne 0 ] && { echo "ABORT_PIPELINE eval R2/R3 v2"; exit 1; }

step "EVAL R4-v2 (recalib)"
CUDA_VISIBLE_DEVICES=0 $EVAL --run-name r4_distill_l2_v2 \
  --checkpoint checkpoints/phase2/r4_distill_l2_v2/final.pt \
  --output results/phase2/eval_r4_distill_l2_v2.json > logs/eval_r4v2.log 2>&1
rc=$?; echo "R4-v2 eval rc=$rc"; [ $rc -ne 0 ] && { echo "ABORT_PIPELINE eval R4-v2"; exit 1; }

echo "PIPELINE_DONE $(date -u +%Y-%m-%dT%H:%M:%S)"
