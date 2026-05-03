# Plan — Shrink TRM ablation grid + paper-aligned architecture cleanup

## Context

We have run the `x_last_only` ablation (TRM head consumes only the VLM's final generation-token hidden state) on three VLMs:

| VLM | TRM dim (= VLM hidden_size) | Best val_acc |
|-----|------------------------------|--------------|
| Qwen3-VL-2B-Instruct | 2048 | **0.282** ← current best |
| Qwen3-VL-4B-Instruct | 2560 | 0.267 |
| Qwen2.5-VL-7B-Instruct | 3584 | 0.252 (still running) |

**User's hypothesis:** monotone dip as TRM dim grows → make the TRM as light as possible.
**My validation of that hypothesis:** the trend is real but currently **confounded** — TRM dim is locked to VLM hidden size, so we can't separate "smaller TRM head" from "less informative VLM features." Adding an input projection lets us run a clean grid.

**Paper-side validation found in `context/...2510.04871v1.pdf`:**

- §4.4 *Less is more*: *"smaller networks are better, but 2 layers seems to be the optimal choice... when data is too scarce and model size is large, there can be an overfitting penalty."*
- §4.5 *Attention-free architecture for tasks with small fixed context length*: *"Self-attention is particularly good for long-context lengths when L ≫ D... However, when focusing on tasks where L ≤ D, a linear layer is too cheap... we can replace the self-attention layer with an MLP."* In x_last_only mode L=3, D∈[2048,3584]; we're firmly in the L ≪ D regime where the paper recommends **MLP-Mixer, no positional encoding** rather than our current self-attention + RoPE.
- §4.8: paper uses **N_sup=16** supervision steps; our code uses 8 (see `experiments/trm_methodology_review.md`).
- §4.5 recommends RoPE *only* for the self-attention variant on long-context grids (Maze-Hard, ARC-AGI 30×30 = 900 tokens). For x_last_only (L=3) RoPE encodes 3 positions — essentially noise.

**Goal:** run a tight grid that varies TRM dim independently of the VLM, drop RoPE for x_last_only, and add the paper-prescribed MLP-Mixer variant on the best dim per VLM.

---

## Code changes

All changes are minimal and gated behind flags so existing runs remain reproducible.

### 1. `trm/model.py`
- **Add input projection.** New `nn.Linear(input_dim, dim, bias=False)` applied to `x` in `TRM.forward`, and to `z` when it comes from the dataset (full-seq mode). `y_init` and `z_init` already live at the TRM dim. Constructor gains `input_dim` arg; defaults to `dim` (no-op when equal).
- **Make RoPE optional.** `TRMBlock(... use_rope: bool = True)`. In `Attention.forward`, skip `apply_rope` when `use_rope=False`. Don't allocate the freq buffers in that case.
- **Add MLP-Mixer block.** New `MixerBlock` class that replaces self-attention with `Linear(L, L) → SwiGLU(D, D*mlp_ratio)` (token-mixing then channel-mixing, paper §4.5). `TRMBlock` gets a `block_type ∈ {"attn", "mixer"}` switch. Mixer requires a fixed L; in x_last_only mode L = 3 (x:1 + y:1 + z:1), in y-update L = 2. Use a single Mixer module that takes per-call seq length and a learnable `Linear(L_max, L_max)` masked to the current L (simplest: two separate mixer blocks, one for L=3 and one for L=2; both shared across recursion steps).

### 2. `trm/dataset.py`
- No change. Dataset already returns x and z at the VLM hidden_dim; the new input projection inside the model handles the dim change.

### 3. `trm/train.py`
- New CLI flags:
  - `--input_dim` (int, default = `dim`) — VLM hidden size; controls the input-projection layer.
  - `--no_rope` (action='store_true').
  - `--block_type` ∈ `{attn, mixer}` (default `attn`).
- Pass these through to `TRM(...)`.
- Default `--n_sup_steps` stays 8 (back-compat); shell scripts will pass `--n_sup_steps 16`.
- MLflow tags: add `input_dim`, `no_rope`, `block_type`.

### 4. New shell script: `scripts/run_xlast_dim_grid.sh`
- Loops dim ∈ {128, 256, 512, 1024, 2048} for **each** VLM (Qwen3-VL-2B at input_dim=2048, Qwen2.5-VL-7B at input_dim=3584).
- Each run uses: `--x_last_only --no_rope --block_type attn --n_sup_steps 16 --no_fp16` (fp32 as established for x_last_only).
- bs=32, grad_accum=2 (eff bs=64), lr=1e-4, patience=20, epochs=50.
- For tiny dims, n_heads is auto-set so head_dim stays ≥ 32: n_heads = max(1, dim // 64).
- Two GPUs in parallel via `&` + `wait`: GPU 0 runs the 2B grid, GPU 1 runs the 7B grid (sequential within each GPU).
- run_tag pattern: `xlast_d{DIM}` (e.g. `xlast_d128`, `xlast_d2048`); per-VLM via the existing `--model_name` argument so MLflow run names stay distinct.

### 5. New shell script: `scripts/run_xlast_mixer.sh`
- Runs once after `run_xlast_dim_grid.sh` finishes.
- Picks the best dim from the grid (we'll inspect MLflow; can be hard-coded after grid completes or parameterized via `BEST_DIM_2B` / `BEST_DIM_7B` env vars).
- Trains MLP-Mixer variant at that dim per VLM: `--block_type mixer --no_rope --x_last_only --n_sup_steps 16`. Other args identical to the grid.

---

## Ablation grid (10 grid runs + 2 mixer follow-ups = 12 total)

| Run | VLM | input_dim | TRM dim | block | RoPE | n_heads | MLP ratio | N_sup |
|-----|-----|-----------|---------|-------|------|---------|-----------|-------|
| 1–5 | Qwen3-VL-2B | 2048 | 128, 256, 512, 1024, 2048 | attn | off | dim/64 | 2 | 16 |
| 6–10 | Qwen2.5-VL-7B | 3584 | 128, 256, 512, 1024, 2048 | attn | off | dim/64 | 2 | 16 |
| 11 | Qwen3-VL-2B | 2048 | best of 1–5 | mixer | off | n/a | 2 | 16 |
| 12 | Qwen2.5-VL-7B | 3584 | best of 6–10 | mixer | off | n/a | 2 | 16 |

Existing runs (`mlp2_xlast` etc.) stay in MLflow as the **with-RoPE, attn, N_sup=8** baseline.

---

## Execution order

0. **Free GPU 1**: kill the in-progress `qwen25_7b_mlp4_xlast` run (`pkill -f "qwen25_7b_mlp4_xlast"` or kill its PID) so the 7B grid has a clean GPU 1. The mlp2 variant on GPU 0 stays.
1. Implement code changes (model.py, train.py, two shell scripts). Sanity-check with one tiny dim (`dim=128`) end-to-end on a few iterations.
2. Launch `bash scripts/run_xlast_dim_grid.sh` — runs for ~hours, two VLMs in parallel.
3. Inspect MLflow once grid completes: `sqlite3 mlflow.db "SELECT r.name, MAX(m.value) FROM runs r JOIN metrics m USING(run_uuid) WHERE m.key='val_acc' AND r.name LIKE '%xlast_d%' GROUP BY r.name;"`
4. Set `BEST_DIM_2B` / `BEST_DIM_7B` and run `bash scripts/run_xlast_mixer.sh`.
5. Final comparison table written into `experiments/trm_dim_grid_results.md`.

---

## Critical files

- [trm/model.py](trm/model.py) — input projection + RoPE flag + MixerBlock
- [trm/train.py](trm/train.py) — new CLI flags, plumb to model, MLflow tags
- [trm/dataset.py](trm/dataset.py) — unchanged
- [scripts/run_xlast_dim_grid.sh](scripts/run_xlast_dim_grid.sh) — new
- [scripts/run_xlast_mixer.sh](scripts/run_xlast_mixer.sh) — new
- [experiments/trm_architecture.md](experiments/trm_architecture.md) — update to document new flags + Mixer variant
- [experiments/trm_dim_grid_results.md](experiments/trm_dim_grid_results.md) — new, written after grid completes

## Reused utilities (no rewriting)

- `EMA`, `TextLogger`, `save_checkpoint` in `trm/utils.py`
- `HiddenStateDataset`, `collate_fn` in `trm/dataset.py` (`x_last_only=True` already supported)
- MLflow setup, system metrics, atomic checkpointing in `trm/train.py`
- Process-launching pattern (`nohup ... &`, `disown`, per-GPU log files) from `scripts/run_qwen25_xlast.sh`

---

## Verification

After implementation, before launching the full grid:

1. **Smoke test**: `python -m trm.train --x_last_only --no_rope --block_type attn --input_dim 2048 --dim 128 --n_heads 2 --epochs 1 --model_name Qwen3-VL-2B-Instruct --gpu 0 ... ` — confirms the new projection + no-RoPE path runs end-to-end on real data and saves a checkpoint.
2. **Mixer smoke test**: same but `--block_type mixer`; verify it forward-passes for both L=3 (z-update) and L=2 (y-update).
3. **Param-count sanity check**: print param count for each grid config; the dim=128 model should be < 1M trainable params (excluding the input projection which is ~0.3–0.5M depending on VLM).
4. **MLflow check after smoke runs**: `sqlite3 mlflow.db "SELECT name, status FROM runs WHERE name LIKE '%xlast_d128%';"` shows the run exists.
5. **End-to-end grid**: confirm 10 grid runs + 2 mixer runs all finish (early-stop or epoch 50). All `best.pt` files exist. MLflow shows monotone or non-monotone dim curve.
6. **Result interpretation**: in `experiments/trm_dim_grid_results.md` plot val_acc vs dim per VLM, plus the mixer points, plus the existing baseline (with-RoPE, attn, N_sup=8). Conclude on "smaller is better" hypothesis and on RoPE/Mixer effect.

## Risks

- **Mixer + variable L:** y-update has L=2, z-update has L=3. Need either two separate mixer modules or one with the largest L padded — I prefer two modules for clarity (paper does the same per-task).
- **Tiny dims (128) may underfit:** that's the point of the grid; we'll see it as the U-curve bottom.
- **Disk:** 12 runs × ~1 GB best.pt = 12 GB. Currently 11 GB free. Will set `--save_optimizer=False` (already default in `save_checkpoint`) and may need to delete intermediate `latest.pt`s mid-grid. Add a periodic `latest.pt` cleanup line to the runner script.
- **GPU 1 conflict resolved by user direction:** kill the running `qwen25_7b_mlp4_xlast` run before launching the 7B grid. The corresponding mlp2 run on GPU 0 stays.
