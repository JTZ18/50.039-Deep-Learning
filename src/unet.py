import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DCBlock(nn.Module):
    """
    A reusable double convolution block used throughout the UNET architecture.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.1):
        super(DCBlock, self).__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.layers(x)

class UNET(nn.Module):
    """Encoder Decoder Convolution Architecture for Semantic Segmentation using UNET."""
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], kernel_size=3, dropout_rate=0.1
    ):
        super(UNET, self).__init__()
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling Encoder for UNET
        for feature in features:
            dcblock = DCBlock(in_channels, feature, kernel_size, dropout_rate=dropout_rate)
            self.downsample.append(dcblock)
            in_channels = feature

        # Upsampling Decoder for UNET
        for feature in reversed(features):
            convTranspose = nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            dcblock = DCBlock(feature*2, feature, kernel_size, dropout_rate=dropout_rate)

            self.upsample.append(convTranspose)
            self.upsample.append(dcblock)

        self.bridge = DCBlock(features[-1], features[-1]*2, kernel_size, dropout_rate=dropout_rate)
        self.conv_out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling Encoder
        for module in self.downsample:
            x = module(x)
            skip_connections.append(x)
            x = self.pooling(x)

        # Bridge from encoder to decoder
        x = self.bridge(x)
        skip_connections = skip_connections[::-1]

        # Upsampling Decoder
        for i in range(0, len(self.upsample), 2):

            # Apply convTranspose upsampling
            x = self.upsample[i](x)
            skip_connection = skip_connections[i // 2]

            # Reshape x to fit skip_connection
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate skip_connection with x
            concat_skip_connection = torch.cat((skip_connection, x), dim=1)

            # Apply double convolution block
            x = self.upsample[i + 1](concat_skip_connection)

        return self.conv_out(x)

def test():
    x = torch.randn(1, 3, 256, 256)
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()