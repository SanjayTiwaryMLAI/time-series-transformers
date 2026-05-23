"""
PatchTST — Patch-based Time Series Transformer
A Time Series is Worth 64 Words (Nie et al., 2022)
"""
import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len  = patch_len
        self.stride     = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.projection  = nn.Linear(patch_len, d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        self.dropout     = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_vars)
        B, L, C = x.shape
        patches  = x.unfold(1, self.patch_len, self.stride)  # (B, num_patches, C, patch_len)
        patches  = patches.mean(dim=2)                        # average over channels
        tokens   = self.projection(patches)                   # (B, num_patches, d_model)
        cls      = self.cls_token.expand(B, -1, -1)
        tokens   = torch.cat([cls, tokens], dim=1)
        return self.dropout(tokens + self.pos_embed)


class PatchTST(nn.Module):
    def __init__(self, seq_len: int = 512, pred_len: int = 96, n_vars: int = 1,
                 d_model: int = 128, num_heads: int = 8, num_layers: int = 3,
                 patch_len: int = 16, stride: int = 8, dropout: float = 0.1):
        super().__init__()
        self.pred_len   = pred_len
        self.patch_emb  = PatchEmbedding(d_model, patch_len, stride, seq_len, dropout)
        encoder_layer   = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model * 4,
                                                     dropout=dropout, batch_first=True)
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head       = nn.Linear(d_model, pred_len * n_vars)
        self.n_vars     = n_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_vars)
        emb = self.patch_emb(x)
        out = self.encoder(emb)
        cls = out[:, 0]                         # CLS token
        return self.head(cls).view(-1, self.pred_len, self.n_vars)


if __name__ == "__main__":
    model = PatchTST(seq_len=512, pred_len=96, n_vars=7)
    x     = torch.randn(4, 512, 7)
    print(f"Input:  {x.shape}")
    print(f"Output: {model(x).shape}")   # (4, 96, 7)
