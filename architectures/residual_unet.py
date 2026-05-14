import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, first_conv_stride, second_conv_stride, dropout_rate=0.0):
        super(FirstConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=first_conv_stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=second_conv_stride,
            padding=1,
        )

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += identity
        return out


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, first_conv_stride, second_conv_stride, dropout_rate=0.0
    ):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels) 
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=first_conv_stride,
            padding=1,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=second_conv_stride,
            padding=1,
        )

        self.skip = nn.Identity()
        if first_conv_stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=first_conv_stride
                ),
                nn.BatchNorm2d(out_channels), 
            )

    def forward(self, x):
        identity = self.skip(x)
        
        out = self.bn1(x)  
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.conv2(out)

        out += identity  
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList(
            [
                FirstConvBlock(in_channels, 64, first_conv_stride=1, second_conv_stride=1, dropout_rate=dropout_rate),
                ResidualBlock(64, 128, first_conv_stride=2, second_conv_stride=1, dropout_rate=dropout_rate),
                ResidualBlock(128, 256, first_conv_stride=2, second_conv_stride=1, dropout_rate=dropout_rate),
            ]
        )

    def forward(self, x):
        skip_connections = []
        for block in self.blocks:
            x = block(x)
            skip_connections.append(x)
        return x, skip_connections


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(Bridge, self).__init__()
        self.block = ResidualBlock(
            in_channels, out_channels, first_conv_stride=2, second_conv_stride=1, dropout_rate=dropout_rate
        )

    def forward(self, x):
        return self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpSampleBlock, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.upconv_blocks = nn.ModuleList(
            [
                UpSampleBlock(in_channels=in_channels, out_channels=256),
                UpSampleBlock(in_channels=256, out_channels=128),
                UpSampleBlock(in_channels=128, out_channels=64),
            ]
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=512,
                    out_channels=256,
                    first_conv_stride=1,
                    second_conv_stride=1,
                    dropout_rate=dropout_rate,
                ),
                ResidualBlock(
                    in_channels=256,
                    out_channels=128,
                    first_conv_stride=1,
                    second_conv_stride=1,
                    dropout_rate=dropout_rate,
                ),
                ResidualBlock(
                    in_channels=128,
                    out_channels=64,
                    first_conv_stride=1,
                    second_conv_stride=1,
                    dropout_rate=dropout_rate,
                ),
            ]
        )

        self.final_conv = FinalConv(in_channels=64, out_channels=out_channels)


    def forward(self, x, skip_connections):
        for upconv, residual_block, skip_connection in zip(
            self.upconv_blocks, self.residual_blocks, skip_connections[::-1]
        ):
            x = upconv(x)
            x = torch.cat([x, skip_connection], dim=1)
            x = residual_block(x)
        
        x = self.final_conv(x)
        return x


class ResidualUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, dropout_rate=0.0):
        super(ResidualUNet, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, dropout_rate=dropout_rate)
        self.bridge = Bridge(in_channels=256, out_channels=512, dropout_rate=dropout_rate)
        self.decoder = Decoder(in_channels=512, out_channels=num_classes, dropout_rate=dropout_rate)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.bridge(x)
        x = self.decoder(x, skip_connections)
        return x



if __name__ == "__main__":
    model = ResidualUNet(in_channels=3)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print("input shape", input.shape)
    print("output shape", output.shape)