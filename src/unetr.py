import torch.nn as nn
import torch
from src.utils.vit import ViT

""" 
    Adapted from: 
        https://github.com/tamasino52/UNETR/blob/main/unetr.py
        https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV 
"""

class YellowBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class BlueBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0), 
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class GreenBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)

class UNETR(nn.Module):
    def __init__(self, img_shape=(256, 256), input_dim=3, output_dim=1, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1):
        """ 
            img_shape: Dimension of image
            input_dim: Number of color channels
            output_dim: Number of output channels
            embed_dim: Embedding dimension
            patch_size: Patch size (should be in multiples of 256)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        self.transformer = ViT (
            input_dim,
            embed_dim,
            img_shape,
            patch_size,
            num_heads,
            self.num_layers,
            dropout
        )

        # NOTE: This follows the U-shape that goes down then up
        self.yellow_encoder_x = \
            nn.Sequential(
                YellowBlock(input_dim, 32),
                YellowBlock(32, 64)
            )
        
        self.blue_encoder_z3 = \
            nn.Sequential(
                BlueBlock(embed_dim, 512),
                BlueBlock(512, 256),
                BlueBlock(256, 128)
            )
        
        self.blue_encoder_z6 = \
            nn.Sequential(
                BlueBlock(embed_dim, 512),
                BlueBlock(512, 256),
            )
        
        self.blue_encoder_z9 = BlueBlock(embed_dim, 512)
        
        self.green_upsampler_z12 = GreenBlock(embed_dim, 512)

        self.yellow_decoder_z9 = \
            nn.Sequential(
                YellowBlock(1024, 512),
                YellowBlock(512, 512),
            )
        self.green_upsampler_z9 = GreenBlock(512, 256)
        
        self.yellow_decoder_z6 = \
            nn.Sequential(
                YellowBlock(512, 256),
                YellowBlock(256, 256),
            )
        self.green_upsampler_z6 = GreenBlock(256, 128)
        
        self.yellow_decoder_z3 = \
            nn.Sequential(
                YellowBlock(256, 128),
                YellowBlock(128, 128),
            )        
        self.green_upsampler_z3 = GreenBlock(128, 64)
        
        self.yellow_decoder_z0 = \
            nn.Sequential(
                YellowBlock(128, 64),
                YellowBlock(64, 64),
            )
        self.final_conv = nn.Conv2d(64, output_dim, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        z3, z6, z9, z12 = self.transformer(x)

        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = self.green_upsampler_z12(z12)

        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = self.blue_encoder_z9(z9)
        z9 = self.yellow_decoder_z9(torch.cat([z9, z12], dim=1))
        z9 = self.green_upsampler_z9(z9)

        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = self.blue_encoder_z6(z6)
        z6 = self.yellow_decoder_z6(torch.cat([z6, z9], dim=1))
        z6 = self.green_upsampler_z6(z6)

        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z3 = self.blue_encoder_z3(z3)
        z3 = self.yellow_decoder_z3(torch.cat([z3, z6], dim=1))
        z3 = self.green_upsampler_z3(z3)

        z0 = self.yellow_encoder_x(x)
        z0 = self.yellow_decoder_z0(torch.cat([z0, z3], dim=1))
        
        output = self.final_conv(z0)
        return output
        
        
