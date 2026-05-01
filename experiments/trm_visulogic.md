# Experiment: TRM on Qwen3-VL Hidden States for VisuLogic

## Hypothesis

The correct answer to visual reasoning MCQ problems is already encoded in the LLM's first forward pass. Autoregressive next-token generation introduces hallucinations and error propagation. A tiny recursive model operating on the frozen hidden states can extract the answer more reliably than the LLM's own generation.

## Method

We extract the final-layer hidden states from Qwen3-VL (2B and 4B Instruct) for each VisuLogic sample, then train a Tiny Recursion Model (TRM) — following "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau) — to classify the answer as A/B/C/D.

## Data

| Split | Samples | Source |
|-------|---------|--------|
| Train | ~4,900 | visulogic_train |
| Val | 999 | visulogic_benchmark |

Hidden states pre-extracted with `scripts/extract_hidden_states.py`. Each tensor is `(1, seq_len, 2048)` from the final transformer layer.

## Mapping Paper Concepts to Our Setup

| Paper (ARC-AGI) | Our Setup |
|---|---|
| x = embedded puzzle | All hidden state tokens except the last (variable length, padded per batch) |
| z = latent reasoning | Hidden state at the last token — the generation prompt token `assistant\n` that feeds into the LM head |
| y = embedded answer | Learnable single-token embedding (1, 1, 2048), initialized randomly |
| Output head | Linear(2048, 4) → A/B/C/D logits |

**Key insight**: z is initialized from the LLM's generation token (not zero), so it starts with compressed reasoning from the full forward pass.

## Architecture Choices

| Choice | Value | Rationale |
|---|---|---|
| Hidden dim | 2048 | Match Qwen hidden size — no projection layer needed, preserves representation fidelity |
| Network layers | 2 | Same as paper |
| Attention heads | 16 (head_dim=128) | Matches Qwen3-VL-2B head structure |
| MLP ratio | 4x (intermediate=8192) | Standard transformer expansion. Reduce to 2x if OOM. |
| Activation | SwiGLU | Same as paper and Qwen |
| Normalization | RMSNorm | Same as paper |
| Positional encoding | RoPE | Same as paper, no bias |
| Estimated params | ~100M | Larger than paper's 7M due to 2048 vs 512 dim, but still tiny vs 2B LLM |

## Recursion Setup

| Parameter | Value | Paper Value | Rationale |
|---|---|---|---|
| n (latent steps) | 6 | 6 | Same — sufficient for reasoning refinement |
| T (deep passes) | 3 | 3 | Same — T-1=2 without grad saves memory |
| N_sup (supervision steps) | 8 | 16 | Reduced — our task is 4-class MCQ, simpler than grid prediction |

## Training Setup

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW (beta1=0.9, beta2=0.95) | Same as paper |
| LR (network) | 1e-4 | Same as paper |
| LR (y_init) | 1e-2 | Paper uses higher LR for embeddings |
| Batch size | 16 | GPU memory constrained (variable seq_len) |
| Effective batch | 64 | 4x gradient accumulation |
| Epochs | 50 | With early stopping (patience=10) |
| LR schedule | Linear warmup (5%) + cosine decay | Standard practice |
| EMA | 0.999 | Same as paper — validation uses EMA weights |
| Weight decay | 0.01 | Standard AdamW |
| Grad clipping | 1.0 | Stability |

## Loss Function

Per deep supervision step:
```
L = CrossEntropy(output_head(y), label) + BCE(q_head(y), correct?)
```

Averaged across all N_sup supervision steps. The halting head learns to predict whether the current answer is correct (ACT mechanism).

## GPU Allocation

- 2x RTX 5060 Ti 16GB
- GPU 0: Train TRM for Qwen3-VL-2B-Instruct
- GPU 1: Train TRM for Qwen3-VL-4B-Instruct
- Both run in parallel

## Logging

- **MLflow**: All metrics per epoch (loss, accuracy, per-tag accuracy, LR)
- **Text logs**: `logs/trm_{model_name}_{timestamp}.txt` — readable via SSH/remote
- **Checkpoints**: `checkpoints/trm/{model_name}/best.pt` and `latest.pt`

## How to Run

```bash
# Both models in parallel
bash scripts/run_trm_train.sh

# Single model
bash scripts/run_trm_train.sh 2B
bash scripts/run_trm_train.sh 4B

# View MLflow dashboard
mlflow ui --port 5000
```

## Baselines to Compare Against

| Method | Description |
|---|---|
| Qwen3-VL-2B direct inference | Autoregressive generation → answer extraction (from eval scripts) |
| Qwen3-VL-4B direct inference | Same |
| Random baseline | 25% (4-class MCQ) |

## What to Watch For

1. **Overfitting**: ~4.9K train, 100M params. EMA + early stopping should help. If overfit is severe, reduce mlp_ratio to 2.
2. **Memory**: Variable seq_len means some batches may be much larger. Dynamic padding handles this, but extreme outliers may cause OOM — reduce batch_size if needed.
3. **Convergence**: If loss doesn't decrease after 5 epochs, check that z initialization from the LLM is actually informative (sanity check: train a linear probe on z alone).
4. **Halting head**: Monitor halt_loss separately — if it doesn't train, the ACT mechanism isn't learning.
