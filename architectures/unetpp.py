import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        batchnorm=True,
        dropout_rate=0.0,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True, dropout_rate=0.0):
        super(Bottleneck, self).__init__()
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x):
        return self.upconv(x)


class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    def __init__(self, in_channels, batchnorm=True, dropout_rate=0.0):
        super(UNetPP, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        # Encoder
        self.conv_0_0 = ConvBlock(
            in_channels=in_channels,
            out_channels=filters[0],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.conv_1_0 = ConvBlock(
            in_channels=filters[0],
            out_channels=filters[1],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.conv_2_0 = ConvBlock(
            in_channels=filters[1],
            out_channels=filters[2],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.conv_3_0 = ConvBlock(
            in_channels=filters[2],
            out_channels=filters[3],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = Bottleneck(
            in_channels=filters[3],
            out_channels=filters[4],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )

        # Upsampling
        self.upconv_1_0 = UpConvBlock(in_channels=filters[1], out_channels=filters[1])
        self.upconv_2_0 = UpConvBlock(in_channels=filters[2], out_channels=filters[2])
        self.upconv_3_0 = UpConvBlock(in_channels=filters[3], out_channels=filters[3])
        self.upconv_bottleneck = UpConvBlock(
            in_channels=filters[4], out_channels=filters[4]
        )

        # Decoder
        self.conv_0_1 = ConvBlock(
            in_channels=filters[0] + filters[1],
            out_channels=filters[0],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.conv_1_1 = ConvBlock(
            in_channels=filters[1] + filters[2],
            out_channels=filters[1],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.conv_2_1 = ConvBlock(
            in_channels=filters[2] + filters[3],
            out_channels=filters[2],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.conv_3_1 = ConvBlock(
            in_channels=filters[3] + filters[4],
            out_channels=filters[3],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )

        self.conv_0_2 = ConvBlock(
            in_channels=filters[0] + filters[0] + filters[1],
            out_channels=filters[0],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.conv_1_2 = ConvBlock(
            in_channels=filters[1] + filters[1] + filters[2],
            out_channels=filters[1],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.conv_2_2 = ConvBlock(
            in_channels=filters[2] + filters[2] + filters[3],
            out_channels=filters[2],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )

        self.conv_0_3 = ConvBlock(
            in_channels=filters[0] + filters[0] + filters[0] + filters[1],
            out_channels=filters[0],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )
        self.conv_1_3 = ConvBlock(
            in_channels=filters[1] + filters[1] + filters[1] + filters[2],
            out_channels=filters[1],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )

        self.conv_0_4 = ConvBlock(
            in_channels=filters[0] + filters[0] + filters[0] + filters[0] + filters[1],
            out_channels=filters[0],
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )

        self.final_conv = FinalConv(in_channels=filters[0], out_channels=1)

    def forward(self, x):
        x_0_0 = self.conv_0_0(x)
        x_1_0 = self.conv_1_0(self.max_pool(x_0_0))
        x_2_0 = self.conv_2_0(self.max_pool(x_1_0))
        x_3_0 = self.conv_3_0(self.max_pool(x_2_0))
        x_bottleneck = self.bottleneck(self.max_pool(x_3_0))

        x_0_1 = self.conv_0_1(torch.cat([x_0_0, self.upconv_1_0(x_1_0)], dim=1))
        x_1_1 = self.conv_1_1(torch.cat([x_1_0, self.upconv_2_0(x_2_0)], dim=1))
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, self.upconv_3_0(x_3_0)], dim=1))
        x_3_1 = self.conv_3_1(
            torch.cat([x_3_0, self.upconv_bottleneck(x_bottleneck)], dim=1)
        )

        x_0_2 = self.conv_0_2(torch.cat([x_0_0, x_0_1, self.upconv_1_0(x_1_1)], dim=1))
        x_1_2 = self.conv_1_2(torch.cat([x_1_0, x_1_1, self.upconv_2_0(x_2_1)], dim=1))
        x_2_2 = self.conv_2_2(torch.cat([x_2_0, x_2_1, self.upconv_3_0(x_3_1)], dim=1))

        x_0_3 = self.conv_0_3(
            torch.cat([x_0_0, x_0_1, x_0_2, self.upconv_1_0(x_1_2)], dim=1)
        )
        x_1_3 = self.conv_1_3(
            torch.cat([x_1_0, x_1_1, x_1_2, self.upconv_2_0(x_2_2)], dim=1)
        )

        x_0_4 = self.conv_0_4(
            torch.cat([x_0_0, x_0_1, x_0_2, x_0_3, self.upconv_1_0(x_1_3)], dim=1)
        )

        return self.final_conv(x_0_4)


if __name__ == "__main__":
    model = UNetPP(in_channels=3)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print("input shape", input.shape)
    print("output shape", output.shape)