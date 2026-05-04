"""
Tiny Recursion Model (TRM) — Self-Attention variant.

Implements the architecture from "Less is More: Recursive Reasoning with Tiny Networks"
adapted for processing frozen LLM hidden states for MCQ classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x_fp32 = x.float()
        norm = x_fp32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * norm).to(dtype) * self.weight


def apply_rope(x, freqs_cos, freqs_sin):
    """Apply rotary positional embeddings."""
    # x: (B, n_heads, seq_len, head_dim)
    seq_len = x.shape[2]
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)  # (1, 1, seq, d//2)
    sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


def precompute_rope(dim, max_len=8192, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim, n_heads, use_rope=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, freqs_cos, freqs_sin)
            k = apply_rope(k, freqs_cos, freqs_sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # mask: (B, S, S) — replace 0s with -inf, clamp NaN after softmax
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        # Zero out attention weights for fully-padded rows to prevent NaN
        attn = attn.nan_to_num(0.0)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, use_rope=True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, use_rope=use_rope)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin, mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MixerLayer(nn.Module):
    """
    MLP-Mixer style block (paper §4.5): token-mixing MLP followed by channel-mixing MLP.
    Recommended when L <= D — our x_last_only mode has L=3 for z-update or L=2 for y-update.
    Both MLPs use SwiGLU activations to match the paper's transformer block style.
    No positional encoding (token positions are absorbed into the token-mix Linear).
    """
    def __init__(self, dim, seq_len, mlp_ratio=4):
        super().__init__()
        self.seq_len = seq_len
        self.norm1 = RMSNorm(dim)
        # Token-mixing: 2-layer SwiGLU MLP applied across the sequence axis.
        # Hidden width = max(seq_len * mlp_ratio, seq_len) so it always has at least L params.
        token_hidden = max(seq_len * mlp_ratio, seq_len)
        self.token_mix = SwiGLU(seq_len, token_hidden)
        self.norm2 = RMSNorm(dim)
        # Channel-mixing: SwiGLU MLP across the feature axis (same as transformer block).
        self.channel_mix = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x, *_args, **_kwargs):
        # x: (B, L, D). Token-mix runs on the L axis -> transpose, mix, transpose back.
        h = self.norm1(x)
        h = self.token_mix(h.transpose(1, 2)).transpose(1, 2)  # (B, L, D)
        x = x + h
        x = x + self.channel_mix(self.norm2(x))
        return x


class TRMBlock(nn.Module):
    """
    The 'tiny network' — a 2-layer Transformer (or MLP-Mixer) used for both z- and y-update.

    For block_type='attn' a single shared block handles any sequence length (L=3 for z, L=2 for y).
    For block_type='mixer', tokens are mixed by Linear(L, L), so we keep two separate mixer
    blocks — one per sequence length — and route to the right one in TRM._update_*.
    """
    def __init__(self, dim=2048, n_heads=16, n_layers=2, mlp_ratio=4,
                 max_seq_len=8192, use_rope=True, block_type='attn', seq_len=None):
        super().__init__()
        self.block_type = block_type
        if block_type == 'attn':
            self.layers = nn.ModuleList([
                TransformerLayer(dim, n_heads, mlp_ratio, use_rope=use_rope)
                for _ in range(n_layers)
            ])
            if use_rope:
                freqs_cos, freqs_sin = precompute_rope(dim // n_heads, max_seq_len)
                self.register_buffer('freqs_cos', freqs_cos)
                self.register_buffer('freqs_sin', freqs_sin)
            else:
                self.freqs_cos = None
                self.freqs_sin = None
        elif block_type == 'mixer':
            assert seq_len is not None, "MixerBlock requires a fixed seq_len"
            self.layers = nn.ModuleList([
                MixerLayer(dim, seq_len, mlp_ratio) for _ in range(n_layers)
            ])
            self.freqs_cos = None
            self.freqs_sin = None
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

    def forward(self, seq, mask=None):
        """
        Args:
            seq: (B, total_seq_len, D) — concatenation of various components
            mask: (B, total_seq_len, total_seq_len) attention mask or None (ignored by Mixer)
        Returns:
            (B, total_seq_len, D)
        """
        for layer in self.layers:
            seq = layer(seq, self.freqs_cos, self.freqs_sin, mask)
        return seq


class TRM(nn.Module):
    """
    Full TRM system: network block + output head + halting head + recursion logic.
    """
    def __init__(self, dim=2048, n_heads=16, n_layers=2, mlp_ratio=4,
                 n_classes=4, n_latent_steps=6, n_deep_passes=3, max_seq_len=8192,
                 input_dim=None, use_rope=True, block_type='attn',
                 z_seq_len=3, y_seq_len=2):
        """
        Args:
            input_dim: VLM hidden size; if different from `dim`, an input projection
                Linear(input_dim, dim) is added so the TRM body can be shrunk
                independently of the VLM dim. Defaults to `dim` (no-op).
            use_rope: whether to apply rotary positional embeddings in attention.
                The paper uses RoPE for long-context self-attention; we disable it
                in x_last_only mode where L=3 makes RoPE essentially noise.
            block_type: 'attn' (default, paper's self-attention variant) or
                'mixer' (paper §4.5 MLP-Mixer for L <= D regime).
            z_seq_len, y_seq_len: required when block_type='mixer'. For our
                x_last_only setup these are 3 (x:1 + y:1 + z:1) and 2 (y:1 + z:1).
        """
        super().__init__()
        self.dim = dim
        self.input_dim = input_dim if input_dim is not None else dim
        self.n_latent_steps = n_latent_steps
        self.n_deep_passes = n_deep_passes
        self.block_type = block_type

        # Optional input projection (VLM hidden dim -> TRM dim).
        if self.input_dim != dim:
            self.input_proj = nn.Linear(self.input_dim, dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        # Tiny network. For attn: one shared block. For mixer: two blocks (one per L).
        if block_type == 'attn':
            self.net = TRMBlock(dim, n_heads, n_layers, mlp_ratio, max_seq_len,
                                use_rope=use_rope, block_type='attn')
            self.net_y = None  # not used in attn mode
        elif block_type == 'mixer':
            self.net = TRMBlock(dim, n_heads, n_layers, mlp_ratio, max_seq_len,
                                use_rope=False, block_type='mixer', seq_len=z_seq_len)
            self.net_y = TRMBlock(dim, n_heads, n_layers, mlp_ratio, max_seq_len,
                                  use_rope=False, block_type='mixer', seq_len=y_seq_len)
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        # Learnable initial y embedding
        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        # Learnable initial z embedding (used when x_last_only ablation overrides
        # the data-provided z; harmless extra param otherwise).
        self.z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Output head: maps y → class logits
        self.output_head = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, n_classes, bias=False),
        )

        # Halting head: maps y → halt probability
        self.q_head = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, 1, bias=False),
        )

    def _build_z_update_mask(self, x_mask, seq_x, seq_y, seq_z, B, device):
        """Build attention mask for z-update: concat(x, y, z)."""
        if x_mask is None:
            return None
        # x_mask: (B, seq_x), extend with ones for y and z
        yz_mask = torch.ones(B, seq_y + seq_z, device=device, dtype=x_mask.dtype)
        full_mask = torch.cat([x_mask, yz_mask], dim=1)  # (B, total_len)
        # (B, 1, total_len) * (B, total_len, 1) → (B, total_len, total_len)
        return full_mask.unsqueeze(1) * full_mask.unsqueeze(2)

    def _update_z(self, x, y, z, x_mask=None):
        """
        z = net(x, y, z): Update latent reasoning.
        Concatenates x, y, z along sequence dim, runs through network,
        extracts the z positions from the output.
        """
        B = x.shape[0]
        combined = torch.cat([x, y, z], dim=1)  # (B, seq_x + 2, D)
        mask = self._build_z_update_mask(
            x_mask, x.shape[1], y.shape[1], z.shape[1], B, x.device)
        out = self.net(combined, mask)
        return out[:, -1:, :]  # last position is z

    def _update_y(self, y, z):
        """
        y = net(y, z): Update prediction using reasoning.
        No x here — only y and z attend to each other.
        """
        combined = torch.cat([y, z], dim=1)  # (B, 2, D)
        # Mixer needs the y-specific block (L=2). Attention can reuse the same block.
        net = self.net_y if (self.block_type == 'mixer') else self.net
        out = net(combined)
        return out[:, :1, :]  # first position is y

    def latent_recursion(self, x, y, z, x_mask=None):
        """Inner loop: n latent steps updating z, then one y update."""
        for _ in range(self.n_latent_steps):
            z = self._update_z(x, y, z, x_mask)
        y = self._update_y(y, z)
        return y, z

    def deep_recursion(self, x, y, z, x_mask=None):
        """
        Outer recursion: T-1 passes without gradients, 1 pass with gradients.
        Returns detached (y, z) for next supervision step, plus logits.
        """
        # T-1 passes without gradients
        with torch.no_grad():
            for _ in range(self.n_deep_passes - 1):
                y, z = self.latent_recursion(x, y, z, x_mask)

        # Final pass with gradients
        y, z = self.latent_recursion(x, y, z, x_mask)

        y_squeezed = y.squeeze(1)  # (B, D)
        pred_logits = self.output_head(y_squeezed)  # (B, n_classes)
        halt_logits = self.q_head(y_squeezed).squeeze(-1)  # (B,)

        return (y.detach(), z.detach()), pred_logits, halt_logits

    def forward(self, x, z, labels=None, x_mask=None, n_sup_steps=8, x_last_only=False):
        """
        Full forward with deep supervision.

        Args:
            x: (B, seq_len, input_dim) — frozen hidden states (all tokens except last)
            z: (B, 1, input_dim) — generation token hidden state (ignored when x_last_only)
            labels: (B,) — ground truth class indices (0-3)
            x_mask: (B, seq_len) — padding mask for x (1=valid, 0=pad)
            n_sup_steps: number of deep supervision steps
            x_last_only: if True, ignore the dataset-provided z and initialise from
                self.z_init (already at TRM dim, no projection).

        Returns:
            dict with losses, predictions, etc.
        """
        B = x.shape[0]
        # Project VLM hidden states to TRM dim (no-op when input_dim == dim).
        x = self.input_proj(x)
        if x_last_only:
            # z is a learnable parameter at TRM dim — no projection needed.
            z = self.z_init.expand(B, -1, -1).to(x.dtype)
        else:
            z = self.input_proj(z)
        y = self.y_init.expand(B, -1, -1)  # (B, 1, D)

        all_logits = []
        all_pred_losses = []
        all_halt_losses = []

        for step in range(n_sup_steps):
            (y, z), pred_logits, halt_logits = self.deep_recursion(x, y, z, x_mask)
            all_logits.append(pred_logits)

            if labels is not None:
                loss_pred = F.cross_entropy(pred_logits, labels)

                with torch.no_grad():
                    pred_correct = (pred_logits.argmax(-1) == labels).float()
                loss_halt = F.binary_cross_entropy_with_logits(halt_logits, pred_correct)

                all_pred_losses.append(loss_pred)
                all_halt_losses.append(loss_halt)

        final_logits = all_logits[-1]

        if labels is not None:
            # Stack and mean — proper gradient flow through each step's loss
            avg_pred_loss = torch.stack(all_pred_losses).mean()
            avg_halt_loss = torch.stack(all_halt_losses).mean()
            avg_loss = avg_pred_loss + avg_halt_loss
        else:
            avg_loss = torch.tensor(0.0, device=x.device)
            avg_pred_loss = torch.tensor(0.0, device=x.device)
            avg_halt_loss = torch.tensor(0.0, device=x.device)

        return {
            'loss': avg_loss,
            'pred_loss': avg_pred_loss,
            'halt_loss': avg_halt_loss,
            'logits': final_logits,
            'all_logits': all_logits,
        }
