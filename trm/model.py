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
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, freqs_cos, freqs_sin)
        k = apply_rope(k, freqs_cos, freqs_sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # mask: (B, S, S) — replace 0s with -inf, clamp NaN after softmax
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        # Zero out attention weights for fully-padded rows to prevent NaN
        attn = attn.nan_to_num(0.0)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin, mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TRMBlock(nn.Module):
    """
    The 'tiny network' — a 2-layer Transformer used for both z-update and y-update.
    Single shared network, as per the paper.
    """
    def __init__(self, dim=2048, n_heads=16, n_layers=2, mlp_ratio=4, max_seq_len=8192):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(dim, n_heads, mlp_ratio) for _ in range(n_layers)
        ])
        freqs_cos, freqs_sin = precompute_rope(dim // n_heads, max_seq_len)
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    def forward(self, seq, mask=None):
        """
        Args:
            seq: (B, total_seq_len, D) — concatenation of various components
            mask: (B, total_seq_len, total_seq_len) attention mask or None
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
                 n_classes=4, n_latent_steps=6, n_deep_passes=3, max_seq_len=8192):
        super().__init__()
        self.dim = dim
        self.n_latent_steps = n_latent_steps
        self.n_deep_passes = n_deep_passes

        # The tiny network (shared for z-update and y-update)
        self.net = TRMBlock(dim, n_heads, n_layers, mlp_ratio, max_seq_len)

        # Learnable initial y embedding
        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

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
        out = self.net(combined)
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

    def forward(self, x, z, labels=None, x_mask=None, n_sup_steps=8):
        """
        Full forward with deep supervision.

        Args:
            x: (B, seq_len, D) — frozen hidden states (all tokens except last)
            z: (B, 1, D) — generation token hidden state
            labels: (B,) — ground truth class indices (0-3)
            x_mask: (B, seq_len) — padding mask for x (1=valid, 0=pad)
            n_sup_steps: number of deep supervision steps

        Returns:
            dict with losses, predictions, etc.
        """
        B = x.shape[0]
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
