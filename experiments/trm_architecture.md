# TRM Architecture — Implementation Notes

A Tiny Recursion Model (TRM) head trained on top of frozen hidden states from a Vision-Language Model (VLM) for VisuLogic 4-way MCQ classification. Adapted from Jolicoeur-Martineau, *"Less is More: Recursive Reasoning with Tiny Networks"*. This document describes what is actually implemented in [trm/](../trm/), not the paper as-written.

## 1. Why TRM Here

VLMs answer MCQ via autoregressive token generation, which can hallucinate or drift. The hypothesis: the answer signal is already present in the **final-layer hidden states** at the moment the model is about to generate. A small recursive head should be able to recover it more reliably than letting the VLM generate text.

We therefore freeze the VLM, extract its final-layer hidden states once per sample, and train a tiny network (~80–130M params depending on dim) that recursively refines a prediction over those frozen states.

## 2. Inputs

For each VisuLogic sample we run the VLM forward once with the chat-templated prompt + image, capture `outputs.hidden_states[-1]` of shape `(1, seq_len, D)`, and save it to disk. `D` and `seq_len` depend on the VLM (see table below).

| VLM | D | head_dim | n_heads (in TRM) | Typical seq_len (median / p99) |
|-----|------|--------|-------|-------------------|
| Qwen3-VL-2B-Instruct | 2048 | 128 | 16 | 263 / 465 |
| Qwen3-VL-4B-Instruct | 2560 | 128 | 20 | 263 / 465 |
| Qwen2.5-VL-7B-Instruct | 3584 | 128 | 28 | varies |

## 3. Two Operating Modes

The dataset (`trm/dataset.py`) supports two ways of feeding hidden states to TRM:

### 3a. **Full-sequence** (default)
```
x = hidden[:-1]   # (seq_len-1, D) — context tokens
z = hidden[-1:]   # (1, D)         — the generation token (LLM's compressed reasoning)
y = learnable y_init parameter (1, D)
```
Optional `max_seq_len` truncates `x` to its tail (preserving the tokens nearest `z`).

### 3b. **x_last_only** (ablation, `--x_last_only`)
```
x = hidden[-1:]                       # (1, D) — only the generation token
z = learnable z_init parameter (1, D) # not from data
y = learnable y_init parameter (1, D)
```
Tests whether the gen-token state alone carries the answer. Side benefit: x is one token, so attention is `(1+1+1)=3` tokens — bs=32+ fits trivially even in fp32.

In both modes, `z` is the "latent reasoning" scratchpad and `y` is the "embedded answer prediction"; the difference is whether `x` is the full context window or just the gen token.

## 4. The Tiny Network

`TRMBlock` ([trm/model.py:104](../trm/model.py#L104)): a 2-layer pre-norm Transformer, shared between z-update and y-update steps.

Per layer:

```
x ← x + Attention(RMSNorm(x))
x ← x + SwiGLU_MLP(RMSNorm(x))
```

| Component | Detail |
|---|---|
| Norm | RMSNorm (computed in fp32, cast back) |
| Attention | Multi-head self-attention with **RoPE** on Q/K, no bias, NaN-clamped softmax for fully-padded rows |
| MLP | SwiGLU, intermediate = `D * mlp_ratio` (`mlp_ratio=2` or `4`) |
| Layers | 2 (paper: 2) |
| Positional | RoPE only (no learned position embeddings, no bias) |
| Activations | bf16/fp16 by default; fp32 when `--no_fp16` |

**Param count** scales as `n_layers * (4·D² (attn) + 3·D·mlp_ratio·D (mlp))` plus the y/z init vectors and heads.

| Config | Approx params |
|---|---|
| D=2048, mlp_ratio=2 | 83.9 M |
| D=2048, mlp_ratio=4 | 134.2 M |
| D=2560, mlp_ratio=2 | 131.1 M |
| D=3584, mlp_ratio=2 | ~260 M |

Heads:
- **`output_head`**: `RMSNorm → Linear(D, 4)` → A/B/C/D logits
- **`q_head`**: `RMSNorm → Linear(D, 1)` → ACT halting logit

Both `y_init` and `z_init` are `nn.Parameter(randn(1,1,D) * 0.02)` and live in their own optimizer group with a higher LR (`lr_embed=1e-2`, vs `lr=1e-4` for the network).

## 5. Recursion

Two nested loops, both happening *inside one forward call* of `TRM.forward`.

### Inner loop — `latent_recursion(x, y, z)` ([trm/model.py:192](../trm/model.py#L192))

```
for _ in range(n_latent_steps):     # n=6
    z ← net(concat[x, y, z])[last]  # z-update sees everything
y ← net(concat[y, z])[first]        # y-update only uses y and z
return y, z
```

`net` is the same 2-layer Transformer block both times — what changes is whether `x` is included in the input.

### Outer loop — `deep_recursion(x, y, z)` ([trm/model.py:199](../trm/model.py#L199))

```
with torch.no_grad():
    for _ in range(n_deep_passes - 1):    # T-1 = 2 passes, no grad
        y, z = latent_recursion(x, y, z)
y, z = latent_recursion(x, y, z)          # final pass, with grad
return (y.detach(), z.detach()),
       output_head(y), q_head(y)
```

Effective network depth per supervision step = `n_latent_steps * n_deep_passes * 2` = 36 layer-equivalents — but only the last `n_latent_steps * 2 = 12` carry gradients. This is the "less is more" trick: emulate a deep network with constant memory.

### Deep supervision — `TRM.forward` ([trm/model.py:218](../trm/model.py#L218))

```
y = y_init.expand(B, ...)
for step in range(n_sup_steps):           # default 8
    (y, z), pred_logits, halt_logits = deep_recursion(x, y, z, x_mask)
    loss_step = CE(pred_logits, label) + BCE(halt_logits, [pred==label])
loss = mean over steps
```

`(y, z)` are detached between steps so each supervision step gets its own gradient path back through one `deep_recursion`. Final prediction = `pred_logits` from the last step (paper also halts early on q>0; we simply average loss across all steps and use the last logits).

## 6. Loss

Per supervision step:
- **Prediction loss**: `CrossEntropy(output_head(y), label)`
- **Halting loss (ACT)**: `BCEWithLogitsLoss(q_head(y), 1{argmax(pred_logits)==label})`

Total loss = mean over `n_sup_steps` of (`pred_loss + halt_loss`). The halt head is informational — it learns when the prediction is already correct — but is not used to early-stop forward passes in our implementation.

## 7. Training Recipe

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.95), wd=0.01 |
| LR (network) | 1e-4 (sqrt-scaled to 5e-5 for full-seq runs with effective bs=16) |
| LR (`y_init`, `z_init`) | 1e-2 |
| Schedule | Linear warmup (5% of total steps) → Cosine decay |
| Grad clip | 1.0 |
| Mixed precision | bf16/fp16 autocast by default; **fp32** for x_last_only ablations (`--no_fp16`) |
| EMA | shadow copy with decay=0.999; **validation runs on EMA weights** |
| Recursion | n=6 latent, T=3 deep, N_sup=8 |
| Early stopping | patience over val_acc (10 or 20 depending on run) |

**Effective batch size:**
- Full-sequence runs: `bs=1, grad_accum=16` → eff 16 (paper used 64)
- x_last_only runs: `bs=32, grad_accum=2` → eff 64 (matches paper)

## 8. Padding Mask

`x` is padded to the max `seq_len` in the batch. The mask is `1` for valid tokens and `0` for padding. `_build_z_update_mask` extends it with `1`s for the appended `y` and `z` positions and turns it into a `(B, L, L)` self-attention mask. Padded rows are zeroed after softmax (`nan_to_num`) to avoid NaN poisoning.

In `x_last_only` mode `seq_x=1` so the mask is effectively all-ones and just there for shape consistency.

## 9. Files

| File | Purpose |
|---|---|
| [trm/model.py](../trm/model.py) | `TRM`, `TRMBlock`, attention, RoPE, SwiGLU |
| [trm/dataset.py](../trm/dataset.py) | `HiddenStateDataset` (full / `x_last_only`), `collate_fn` |
| [trm/train.py](../trm/train.py) | training loop, mlflow logging, EMA, checkpointing |
| [trm/utils.py](../trm/utils.py) | `EMA`, `TextLogger`, `save_checkpoint` (atomic, optimizer optional) |
| [scripts/extract_hidden_states_v2.py](../scripts/extract_hidden_states_v2.py) | one-time VLM forward pass to dump hidden states (`--last_only` saves only the generation token) |
| [scripts/run_trm_train.sh](../scripts/run_trm_train.sh) | parallel ablation launcher (Qwen3-VL-2B) |
| [scripts/run_qwen25_xlast.sh](../scripts/run_qwen25_xlast.sh) | extract + cleanup-cache + parallel mlp2/mlp4 train (Qwen2.5-VL-7B) |

## 10. Logging & Checkpoints

- **MLflow** (sqlite at `./mlflow.db`, view with `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`):
  - Per-iteration: `iter/train_loss`, `iter/lr`, `iter/epoch` (every 20 optimizer steps)
  - Per-epoch: `train_*`, `val_*`, `val_tag_<tag>` (per-category accuracy), `lr`, `best_val_acc`
  - System metrics: `system/cpu_*`, `system/gpu_*`, `system/system_memory_*` (every 10s)
  - Tags: `model_name`, `mlp_ratio`, `max_seq_len`, `x_last_only`, `ablation`
- **Checkpoints** under `checkpoints/trm/<model>/<run_tag>/`:
  - `best.pt` — model + EMA weights (no optimizer state to save disk)
  - `latest.pt` — same, overwritten each epoch
  - All saves are atomic (`<path>.tmp` + `os.replace`).

## 11. Open Questions / Follow-ups

- **Does `z` initialization matter?** Full-seq uses `z = hidden[-1]` (LLM gen token). x_last_only initializes `z` from a learnable parameter and uses the gen token as `x`. Is one strictly better?
- **Does deep supervision help here?** Setting `n_sup_steps=1` would test whether the recursive structure is doing real work or whether the gain is just from the larger effective compute graph.
- **EMA vs. raw weights at val:** we always use EMA for val. Worth a sanity check that EMA helps on this small dataset.
- **Halting head usage:** currently informational. Could be wired into early-stop at inference for a tiny speedup at the cost of accuracy.
