import torch.nn as nn
import torch.nn.functional as F
import torch
from einops.layers.torch import Rearrange

"""
    Adapted from:
        https://github.com/tamasino52/UNETR/blob/main/unetr.py
        https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV
"""

class Embeddings(nn.Module):
    """ Convolutional positional embeddings for the image """
    def __init__(self, input_dim, embed_dim, img_size, patch_size, dropout):
        """
            input_dim: Number of color channels
            embed_dim: Embedding dimension
            img_size: Dimension of image
            patch_size: Patch size
            dropout: Dropout rate
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = int((img_size[0] * img_size[1]) / (patch_size * patch_size))
        self.patch_embeddings = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    """ Self-attention Block """
    def __init__(self, embed_dim, num_heads, dropout):
        """
            num_heads: Number of attention heads
            embed_dim: Embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.input_rearrange(self.qkv(x))
        q, k, v = qkv[0], qkv[1], qkv[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * (self.head_dim**-0.5))
        att_mat = self.attn_dropout(self.softmax(att_mat))
        x = torch.einsum("blxy,blyd->blxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.proj_dropout(self.out_proj(x))
        return x


class PositionwiseFeedForward(nn.Module):
    "" "Implements FFN equation, based on original paper """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class TransformerBlock(nn.Module):
    """ Transformer Block """
    def __init__(self, embed_dim, num_heads, dropout):
        """
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = SelfAttention(embed_dim, num_heads, dropout)

        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = PositionwiseFeedForward(embed_dim, 2048, dropout)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class ViT(nn.Module):
    """ Vision Transformer modified to extract hidden states from specified layers """
    def __init__(self, input_dim, embed_dim, img_size, patch_size, num_heads, num_layers, dropout):
        """
            NOTE: img_size must be a multiple of patch_size
            NOTE: embed_dim must be a multiple of num_heads

            input_dim: Number of color channels
            embed_dim: Embedding dimension
            img_size: Dimension of image
            patch_size: Patch size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, img_size, patch_size, dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, dropout)
                for i in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.embeddings(x)
        extract_layers = []

        for block in self.blocks:
            x = block(x)
            extract_layers.append(x)

        x = self.norm(x)
        return extract_layers[2], extract_layers[5], extract_layers[8], x
