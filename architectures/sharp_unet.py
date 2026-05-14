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
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)


class SharpBlock(nn.Module):
    def __init__(self, channels):
        super(SharpBlock, self).__init__()

        # Kernel Laplaciano
        laplacian_filter = (
            torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )  

        self.register_buffer(
            "laplacian_filter", laplacian_filter.repeat(channels, 1, 1, 1)
        )

        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels, 
            bias=False,
        )

        with torch.no_grad():
            self.depthwise_conv.weight.copy_(self.laplacian_filter)

        self.depthwise_conv.weight.requires_grad = False  

    def forward(self, x):
        return self.depthwise_conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, batchnorm=True, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels, 64, batchnorm=batchnorm, dropout_rate=dropout_rate
                ),
                ConvBlock(64, 128, batchnorm=batchnorm, dropout_rate=dropout_rate),
                ConvBlock(128, 256, batchnorm=batchnorm, dropout_rate=dropout_rate),
                ConvBlock(256, 512, batchnorm=batchnorm, dropout_rate=dropout_rate),
            ]
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            skip_connections.append(x)
            x = self.max_pool(x)
        return x, skip_connections


class Bottleneck(nn.Module):
    def __init__(self, batchnorm=True, dropout_rate=0.0):
        super(Bottleneck, self).__init__()
        self.conv = ConvBlock(512, 1024, batchnorm=batchnorm, dropout_rate=dropout_rate)

    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.upconv(x)


class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, batchnorm=True, dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.upconv_blocks = nn.ModuleList(
            [
                UpConvBlock(1024, 512),
                UpConvBlock(512, 256),
                UpConvBlock(256, 128),
                UpConvBlock(128, 64),
            ]
        )
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(1024, 512, batchnorm=batchnorm, dropout_rate=dropout_rate),
                ConvBlock(512, 256, batchnorm=batchnorm, dropout_rate=dropout_rate),
                ConvBlock(256, 128, batchnorm=batchnorm, dropout_rate=dropout_rate),
                ConvBlock(128, 64, batchnorm=batchnorm, dropout_rate=dropout_rate),
            ]
        )
        self.sharp_blocks = nn.ModuleList(
            [
                SharpBlock(512),
                SharpBlock(256),
                SharpBlock(128),
                SharpBlock(64),
            ]
        )
        self.final_conv = FinalConv(64, 1)

    def forward(self, x, skip_connections):
        for upconv_block, conv_block, sharp_block, skip_connection in zip(
            self.upconv_blocks,
            self.conv_blocks,
            self.sharp_blocks,
            skip_connections[::-1],
        ):
            x = upconv_block(x)
            sharp_features = sharp_block(skip_connection)  
            x = torch.cat((x, sharp_features), dim=1)
            x = conv_block(x)

        x = self.final_conv(x)
        return x


class SharpUNet(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.0, batchnorm=True):
        super(SharpUNet, self).__init__()
        self.encoder = Encoder(
            in_channels, batchnorm=batchnorm, dropout_rate=dropout_rate
        )
        self.bottleneck = Bottleneck(batchnorm=batchnorm, dropout_rate=dropout_rate)
        self.decoder = Decoder(batchnorm=batchnorm, dropout_rate=dropout_rate)

    def forward(self, x):
        output, skip_connections = self.encoder(x)
        output = self.bottleneck(output)
        output = self.decoder(output, skip_connections)
        return output


if __name__ == "__main__":
    model = SharpUNet(in_channels=3)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print("input shape", input.shape)
    print("output shape", output.shape)