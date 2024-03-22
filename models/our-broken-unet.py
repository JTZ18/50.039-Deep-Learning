class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Unet, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = self.double_convolution(in_channels, 64)
        self.down_convolution_2 = self.double_convolution(64, 128)
        self.down_convolution_3 = self.double_convolution(128, 256)
        self.down_convolution_4 = self.double_convolution(256, 512)
        self.down_convolution_5 = self.double_convolution(512, 1024)

        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,kernel_size=2,stride=2)

        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = self.double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256,kernel_size=2,stride=2)
        self.up_convolution_2 = self.double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,kernel_size=2,stride=2)
        self.up_convolution_3 = self.double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=2,stride=2)
        self.up_convolution_4 = self.double_convolution(128, 64)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, out_channels=out_channels,
            kernel_size=1
        )

    def double_convolution(self, in_channels, out_channels):
        conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv_op

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)
        # *** DO NOT APPLY MAX POOL TO down_9 ***

        up_1 = self.up_transpose_1(down_9)
        if up_1.shape != down_7.shape:
            up_1 = torch.nn.functional.interpolate(up_1, size=down_7.shape[2:], mode='bilinear', align_corners=True)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))

        up_2 = self.up_transpose_2(x)
        if up_2.shape != down_5.shape:
            up_2 = torch.nn.functional.interpolate(up_2, size=down_5.shape[2:], mode='bilinear', align_corners=True)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))

        up_3 = self.up_transpose_3(x)
        if up_3.shape != down_3.shape:
            up_3 = torch.nn.functional.interpolate(up_3, size=down_3.shape[2:], mode='bilinear', align_corners=True)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))

        up_4 = self.up_transpose_4(x)
        if up_4.shape != down_1.shape:
            up_4 = torch.nn.functional.interpolate(up_4, size=down_1.shape[2:], mode='bilinear', align_corners=True)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))

        out = self.out(x)
        return out

        # # Encoder
        # s1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)(x)
        # s1 = torch.nn.ReLU()(s1)
        # s1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)(s1)
        # s1 = torch.nn.ReLU()(s1)
        # p1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(s1)

        # s2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)(p1)
        # s2 = torch.nn.ReLU()(s2)
        # s2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)(s2)
        # s2 = torch.nn.ReLU()(s2)
        # p2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(s2)

        # s3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)(p2)
        # s3 = torch.nn.ReLU()(s3)
        # s3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)(s3)
        # s3 = torch.nn.ReLU()(s3)
        # p3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(s3)

        # s4 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)(p3)
        # s4 = torch.nn.ReLU()(s4)
        # s4 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)(s4)
        # s4 = torch.nn.ReLU()(s4)
        # p4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(s4)

        # # bridge
        # b1 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)(p4)
        # b1 = torch.nn.ReLU()(b1)
        # b1 = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)(b1)
        # b1 = torch.nn.ReLU()(b1)

        # # Decoder
        # d1 = torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)(b1)
        # d1 = torch.nn.ReLU()(d1)
        # d1 = torch.cat([d1, s4], dim=1)
        # d1 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)(d1)
        # d1 = torch.nn.ReLU()(d1)
        # d1 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)(d1)
        # d1 = torch.nn.ReLU()(d1)

        # d2 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)(d1)
        # d2 = torch.nn.ReLU()(d2)
        # d2 = torch.cat([d2, s3], dim=1)
        # d2 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)(d2)
        # d2 = torch.nn.ReLU()(d2)
        # d2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)(d2)
        # d2 = torch.nn.ReLU()(d2)

        # d3 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)(d2)
        # d3 = torch.nn.ReLU()(d3)
        # d3 = torch.cat([d3, s2], dim=1)
        # d3 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)(d3)
        # d3 = torch.nn.ReLU()(d3)
        # d3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)(d3)
        # d3 = torch.nn.ReLU()(d3)

        # d4 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)(d3)
        # d4 = torch.nn.ReLU()(d4)
        # d4 = torch.cat([d4, s1], dim=1)
        # d4 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)(d4)
        # d4 = torch.nn.ReLU()(d4)
        # d4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)(d4)
        # d4 = torch.nn.ReLU()(d4)
        # d4 = torch.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1)(d4)
        # d4 = torch.nn.ReLU()(d4)
        # d4 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0)(d4)
        # d4 = torch.nn.ReLU()(d4)
        # d4 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0)(d4)
        # out = torch.nn.Sigmoid()(d4)

        # return out