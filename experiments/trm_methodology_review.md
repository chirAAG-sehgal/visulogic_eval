# TRM Implementation: Methodology Review & Verification

## 1. What We Are Building

A **Tiny Recursion Model (TRM)** that operates on the frozen final-layer hidden states of Qwen3-VL (2B and 4B Instruct) to classify VisuLogic MCQ answers (A/B/C/D). The core hypothesis is that the correct answer is already encoded in the LLM's first forward pass, and recursive refinement on frozen representations can extract it more reliably than autoregressive generation.

### Paper-to-Implementation Mapping

| Paper Concept | Our Implementation | Correct? |
|---|---|---|
| x = embedded puzzle | All hidden state tokens except the last (variable seq_len, padded per batch) | YES |
| z = latent reasoning | Last hidden state token (generation prompt token, i.e. the token right before the LM head would produce output) | YES — this is a strong design choice: z starts with the LLM's compressed reasoning rather than zeros |
| y = embedded answer | Learnable `(1, 1, D)` parameter, initialized N(0, 0.02) | YES — paper's y_init is also learnable |
| net(x,y,z) → z | Concat [x, y, z], run through shared 2-layer transformer, extract last position | YES |
| net(y,z) → y | Concat [y, z], run through same transformer (no x), extract first position | YES |
| Output head | RMSNorm → Linear(D, 4) → A/B/C/D logits | YES |
| Halting head (ACT) | RMSNorm → Linear(D, 1) → halt probability | YES |

---

## 2. Architecture Choices Verification

| Parameter | Our Value | Paper Value | Verdict |
|---|---|---|---|
| Network layers | 2 | 2 | CORRECT |
| Activation | SwiGLU | SwiGLU | CORRECT |
| Normalization | RMSNorm | RMSNorm | CORRECT |
| Positional encoding | RoPE, no bias | RoPE, no bias | CORRECT |
| Hidden dim (D) | 2048 | 512 | JUSTIFIED — matches Qwen3-VL-2B hidden size; avoids lossy projection layer |
| Attention heads | 16 (head_dim=128) | N/A (paper uses 512/?) | REASONABLE — matches Qwen3-VL-2B head structure |
| MLP ratio | 4x | N/A | STANDARD — paper uses standard expansion |
| Estimated params | ~100M | ~7M | EXPECTED — scales with D^2 (2048² vs 512²); still <5% of the 2B LLM |
| n (latent steps) | 6 | 6 | CORRECT |
| T (deep passes) | 3 | 3 | CORRECT |
| N_sup (supervision) | 8 | 16 | JUSTIFIED — 4-class MCQ is simpler than grid prediction |

### ISSUE: dim=2048 used for BOTH 2B and 4B models

The `run_trm_train.sh` uses `--dim 2048` for both Qwen3-VL-2B and Qwen3-VL-4B. This is correct for the 2B model (Qwen3-VL-2B-Instruct hidden_size=2048), but Qwen3-VL-4B-Instruct likely has a different hidden dimension (e.g. 2560 or 3584). The pre-extracted `.pt` tensors would have shape `(1, seq_len, hidden_dim_of_4B)`, and the TRM would fail at runtime if there's a mismatch.

**Action needed**: Verify the actual hidden size of Qwen3-VL-4B-Instruct and set `--dim` accordingly in the shell script.

---

## 3. Training Setup Verification

| Parameter | Our Value | Paper Value | Verdict |
|---|---|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95) | AdamW (β1=0.9, β2=0.95) | CORRECT |
| LR (network) | 1e-4 | 1e-4 | CORRECT |
| LR (y_init embedding) | 1e-2 | 1e-2 | CORRECT |
| EMA decay | 0.999 | 0.999 | CORRECT |
| Weight decay | 0.01 | N/A | STANDARD |
| Batch size | 16 (effective 64 via grad accum) | 768 | MUCH SMALLER — constrained by variable seq_len and GPU memory (16GB RTX 5060 Ti). Paper operates on fixed 30x30 grids. Acceptable given our setup. |
| Grad clipping | 1.0 | N/A | STANDARD — good for stability |
| LR schedule | Linear warmup (5%) + cosine | N/A | STANDARD |
| Early stopping | patience=10 | N/A | GOOD — necessary given ~4.9K train samples and ~100M params (overfitting risk) |

---

## 4. Review of Last Commit (`2f23c98 "first trainin"`)

The commit introduced the entire TRM implementation from scratch. Key design decisions:

### 4.1 Dataset Design (`trm/dataset.py`)
- **Split hidden state into x and z**: `x = hidden[:-1]` (all tokens except last), `z = hidden[-1:]` (last token). This maps directly to the paper's x/z split.
- **Custom collate with dynamic padding**: Variable-length x is padded with zeros per batch, with a binary mask.
- **Label mapping**: A→0, B→1, C→2, D→3.
- **Verdict**: CORRECT. Clean and faithful to the paper's data model.

### 4.2 Model Architecture (`trm/model.py`)
- **Single shared TRMBlock** for both z-update and y-update — exactly as the paper specifies.
- **Pre-norm architecture**: `Input → Norm → Attention → Add → Norm → MLP → Add` — standard pre-norm transformer, matches paper.
- **RoPE buffers**: Precomputed cos/sin registered as non-trainable buffers — correct.
- **y_init**: Learnable parameter `N(0, 0.02)` — correct scale.
- **Deep recursion**: T-1 passes under `torch.no_grad()`, final pass with gradients. y and z detached between supervision steps. Exactly as specified.
- **Verdict**: CORRECT implementation of the paper's algorithm.

### 4.3 Training Script (`trm/train.py`)
- **Separate LR groups**: y_init at 1e-2, everything else at 1e-4 — matches paper.
- **EMA validation**: Validates using `ema.shadow` — correct, paper uses EMA for evaluation.
- **Per-tag accuracy**: Tracks accuracy by reasoning category — useful diagnostic.
- **MLflow + text logging**: Dual logging system.
- **Verdict**: CORRECT overall. Some issues fixed in uncommitted changes (see below).

### 4.4 Issues in the Committed Code
1. **Loss accumulation via running sum**: `total_loss = total_loss + loss_pred` where `total_loss` starts as `torch.tensor(0.0)`. Works but creates an unnecessary computation chain.
2. **No mixed precision**: All computation in fp32 — works but slow on 16GB GPUs.
3. **Masking uses `float('-inf')`**: Can produce NaN when an entire row is padded (softmax of all -inf = NaN).
4. **RoPE dtype mismatch**: RoPE buffers are fp32 but input could be fp16 — no `.to(x.dtype)` call.
5. **Scheduler step counting**: Integer division `len(loader) * epochs // grad_accum_steps` loses steps vs the actual stepping logic.
6. **No `drop_last`**: Last partial batch could create noisy gradients under gradient accumulation.
7. **`compute_accuracy` imported but unused** (from utils).

---

## 5. Review of Uncommitted Changes

The uncommitted changes address the issues above. Each change reviewed:

### 5.1 `model.py` Changes

| Change | Correct? | Reasoning |
|---|---|---|
| RMSNorm: explicit `dtype = x.dtype` variable | YES | Cleaner, functionally identical |
| apply_rope: `.to(x.dtype)` on cos/sin | YES — CRITICAL | Necessary for fp16 mixed precision. Without this, fp32 buffers * fp16 activations would fail or silently upcast |
| Attention: `float('-inf')` → `-1e9` | YES | `-1e9` is safer in fp16 (representable range). Also avoids all-inf rows producing NaN |
| Attention: `attn.nan_to_num(0.0)` after softmax | YES | Catches any remaining NaN from fully-padded rows. Defensive and correct — padded positions contribute zero attention |
| `_build_z_update_mask` extracted to method | YES | Code reuse/readability. No behavioral change |
| `_update_z` / `_update_y` simplified | YES | Removed intermediate variables. No behavioral change |
| Loss: `torch.stack(losses).mean()` instead of running sum / n | YES | Functionally equivalent but cleaner. `torch.stack().mean()` is more idiomatic PyTorch and avoids ambiguity about the initial `torch.tensor(0.0)` participating in the computation graph |

### 5.2 `train.py` Changes

| Change | Correct? | Reasoning |
|---|---|---|
| Mixed precision (`autocast` + `GradScaler`) | YES | Essential for 16GB GPUs with ~100M param model and variable seq_len. `scaler.unscale_()` before grad clipping is correct protocol |
| `math.ceil(len(loader) / grad_accum_steps)` | YES | Matches the actual stepping logic: `if (step+1) % accum == 0 or (step+1) == len(loader)`. Integer division undercounts |
| `warmup_steps = max(1, ...)` and `T_max = max(1, ...)` | YES | Prevents zero-division in schedulers |
| `drop_last=True` on train loader | YES | Prevents partial batches from creating disproportionate gradient noise. Val loader correctly keeps all samples |
| Empty dataset guard | YES | Fast-fail with clear error message |
| `torch.cuda.manual_seed_all` | YES | Seeds all GPUs, more robust than `manual_seed` for single GPU |
| `.detach()` on logits before accuracy | YES | Prevents unnecessary graph retention during metric computation |
| `try/finally` around training loop | YES | Ensures `mlflow.end_run()` is called even on crash. Previously a crash would leave the MLflow run open |
| Patience counter logging | YES | Better UX — shows progress toward early stopping |
| Removed unused `compute_accuracy` import | YES | Cleanup |
| `run = mlflow.start_run(...)` captures return | NEUTRAL | Not used, but harmless |

### 5.3 All uncommitted changes are VERIFIED CORRECT

No change introduces a regression. All are either bug fixes (RoPE dtype, NaN masking, scheduler counting) or justified improvements (mixed precision, better error handling).

---

## 6. Potential Risks & Recommendations

### 6.1 Overfitting Risk (HIGH)
~100M parameters trained on ~4,900 samples. Mitigations in place: EMA (0.999), early stopping (patience=10), weight decay (0.01). **Recommendation**: Monitor train vs val accuracy divergence. If severe overfitting, reduce `mlp_ratio` from 4 to 2 (cuts params by ~40%).

### 6.2 dim Mismatch for 4B Model (ACTION REQUIRED)
The shell script uses `--dim 2048` for both models. Must verify Qwen3-VL-4B hidden dimension and update. A wrong dim will cause a runtime crash when loading `.pt` tensors of different width.

### 6.3 Effective Batch Size vs Paper
Paper uses batch_size=768, we use effective 64 (16 × 4 accumulation). This is 12x smaller, which means:
- Noisier gradients (partially offset by grad clipping)
- Potentially different convergence dynamics
- The cosine LR schedule may need tuning

### 6.4 No Data Augmentation
Paper uses color permutation, dihedral-group transforms, and translations for ARC-AGI. We have no augmentation on hidden states. This is reasonable — augmenting frozen LLM representations is non-trivial and not directly applicable.

### 6.5 No ACT Early Stopping at Inference
The halting head is trained but not used for early stopping during inference (the model always runs all N_sup steps). This is fine for initial experiments but could be added later for efficiency.

---

## 7. Summary

| Aspect | Status |
|---|---|
| Paper algorithm faithfulness | CORRECT — all core components match |
| Architecture choices | JUSTIFIED — dim scaled to match Qwen hidden size |
| Last commit code | FUNCTIONAL but had fp16/NaN/scheduler bugs |
| Uncommitted changes | ALL CORRECT — fix real bugs, add mixed precision |
| Ready to train? | YES for 2B model; VERIFY dim for 4B model first |
