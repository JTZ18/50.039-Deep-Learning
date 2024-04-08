import torch.nn as nn
import torch
from src.utils.vit import ViT

""" Adapted from: https://paperswithcode.com/method/unetr """

class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0), 
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

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
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        self.transformer = ViT (
            input_dim,
            embed_dim,
            img_shape,
            patch_size,
            num_heads,
            self.num_layers,
            dropout,
            self.ext_layers
        )

        # Very top, just connecting the original with output
        self.yellow_decoder_x = \
            nn.Sequential(
                Conv2DBlock(input_dim, 32, 3),
                Conv2DBlock(32, 64, 3)
            )
        
        self.blue_decoder_z3 = \
            nn.Sequential(
                Deconv2DBlock(embed_dim, 512),
                Deconv2DBlock(512, 256),
                Deconv2DBlock(256, 128)
            )
        
        self.blue_decoder_z6 = \
            nn.Sequential(
                Deconv2DBlock(embed_dim, 512),
                Deconv2DBlock(512, 256),
            )
        
        self.blue_decoder_z9 = Deconv2DBlock(embed_dim, 512)
        
        self.green_upsampler_z3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.green_upsampler_z6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.green_upsampler_z9 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.green_upsampler_z12 = nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2, padding=0, output_padding=0)

        self.yellow_decoder_z0 = \
            nn.Sequential(
                Conv2DBlock(128, 64),
                Conv2DBlock(64, 64),
            )
        self.yellow_decoder_z3 = \
            nn.Sequential(
                Conv2DBlock(256, 128),
                Conv2DBlock(128, 128),
            )
        self.yellow_decoder_z6 = \
            nn.Sequential(
                Conv2DBlock(512, 256),
                Conv2DBlock(256, 256),
            )
        self.yellow_decoder_z9 = \
            nn.Sequential(
                Conv2DBlock(1024, 512),
                Conv2DBlock(512, 512),
            )

        self.final_conv = nn.Conv2d(64, output_dim, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.green_upsampler_z12(z12)

        z9 = self.blue_decoder_z9(z9)
        z9 = self.yellow_decoder_z9(torch.cat([z9, z12], dim=1))
        z9 = self.green_upsampler_z9(z9)

        z6 = self.blue_decoder_z6(z6)
        z6 = self.yellow_decoder_z6(torch.cat([z6, z9], dim=1))
        z6 = self.green_upsampler_z6(z6)

        z3 = self.blue_decoder_z3(z3)
        z3 = self.yellow_decoder_z3(torch.cat([z3, z6], dim=1))
        z3 = self.green_upsampler_z3(z3)

        z0 = self.yellow_decoder_x(z0)
        z0 = self.yellow_decoder_z0(torch.cat([z0, z3], dim=1))
        
        output = self.final_conv(z0)
        return output
        
        
