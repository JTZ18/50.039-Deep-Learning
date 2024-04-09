import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth
from typing import List, Iterable

class LayerNorm2d(nn.Module):
    """ Layer Norm with channel swapping and support """
    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.ln(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
class ResidualAdd(nn.Module):
    """Just an util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class OverlapPatchMerging(nn.Module):
    """ Overlapping patch merging layer """
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class EfficientMultiHeadAttention(nn.Module):
    """ Efficient Multi Head Attention Layer """
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio),
            LayerNorm2d(channels),
        )
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)

        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.attn(x, reduced_x, reduced_x)[0]

        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out

class MixFNN(nn.Module):
    """ Mix MLP Layer """
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        self.mlp1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv2d(
            channels,
            channels * expansion,
            kernel_size=3,
            groups=channels,
            padding=1,
        )
        self.gelu = nn.GELU()
        self.mlp2 = nn.Conv2d(channels * expansion, channels, kernel_size=1)
    
    def forward(self, x):
        x = self.mlp1(x)
        x = self.conv3(x)
        x = self.gelu(x)
        x = self.mlp2(x)
        return x

class MiT(nn.Module):
    """ MiT Block includign Efficient Self-Attention and Mix-FFN """
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        dropout: float = .0
    ):
        super().__init__()
        self.attn = nn.Sequential(
            LayerNorm2d(channels),
            EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
            nn.Dropout(dropout)
        )

        self.mix = nn.Sequential(
            LayerNorm2d(channels),
            MixFNN(channels, expansion=mlp_expansion),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mix(x)
        return x

class TransformerBlock(nn.Module):
    """ Orange Transformer Block """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        dropout: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(in_channels, out_channels, patch_size, overlap_size)
        self.mit = nn.ModuleList(
            [
                MiT(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, dropout[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x):
        x = self.overlap_patch_merge(x)
        for mit in self.mit:
            x = mit(x)
        x = self.norm(x)
        return x

def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk

class Encoder(nn.Module):
    """ Encoder (Front) portion """
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        dropout: float = .0
    ):
        super().__init__()
        dropout =  [x.item() for x in torch.linspace(0, dropout, sum(depths))]
        self.stages = nn.ModuleList(
            [
                TransformerBlock(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(dropout, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class MLPLayer(nn.Module):
    """ Gray MLPLayer in decoder """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.mlp = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        return self.upsample(self.mlp(x))

class Decoder(nn.Module):
    """ Decoder (Back) portion """
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                MLPLayer(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )

    def forward(self, features):
        new_features = []
        for feature, stage in zip(features, self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features

class SegmentationHead(nn.Module):
    """ Output segmentation head, blue MLP at the back """
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        # Fuse 4 channels into 1
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(channels)
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fusion(x)
        x = self.predict(x)
        return x

class SegFormer(nn.Module):
    """ SegFormer """
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: int,
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0,
    ):
        """
            in_channels: Number of input channels
            widths: Widths of the encoder stages
            depths: Depth of each encoder stage
            all_num_heads: Number of heads in each stage
            patch_sizes: Patch sizes in each stage
            overlap_sizes: Overlap sizes in each stage
            reduction_ratios: Reduction ratios in each stage
            mlp_expansions: MLP expansion ratios in each stage
            decoder_channels: Decoder channels
            scale_factors: Scale factors in the decoder
            num_classes: Number of classes
            drop_prob: Dropout probability
        """
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = Decoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegmentationHead(decoder_channels, num_classes, num_features=len(widths))

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)

        # Interpolate to match the size of the input
        segmentation = torch.nn.functional.interpolate(segmentation, size=x.size()[2:], mode='bilinear', align_corners=False)
        return segmentation


segformer = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=1,
)

segmentation = segformer(torch.randn((1, 3, 224, 224)))
print(segmentation.shape) # torch.Size([1, 1, 224, 224])