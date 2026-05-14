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


class Encoder(nn.Module):
    def __init__(self, in_channels, batchnorm, dropout_rate):
        super(Encoder, self).__init__()
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=64,
                    batchnorm=batchnorm,
                    dropout_rate=dropout_rate,
                ),
                ConvBlock(
                    in_channels=64,
                    out_channels=128,
                    batchnorm=batchnorm,
                    dropout_rate=dropout_rate,
                ),
                ConvBlock(
                    in_channels=128,
                    out_channels=256,
                    batchnorm=batchnorm,
                    dropout_rate=dropout_rate,
                )
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
    def __init__(self, batchnorm, dropout_rate):
        super(Bottleneck, self).__init__()
        self.conv = ConvBlock(
            in_channels=256,
            out_channels=512,
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


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  
            nn.BatchNorm2d(F_int),  
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.BatchNorm2d(F_int),  
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),  
            nn.BatchNorm2d(1),
            nn.Sigmoid(), 
        )

        self.relu = (nn.ReLU())

    def forward(self, g, skip_connection):
        g1 = self.W_g(g) 
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return (skip_connection * psi)
    

class Decoder(nn.Module):
    def __init__(self, batchnorm, dropout_rate):
        super(Decoder, self).__init__()
        self.upconv_blocks = nn.ModuleList(
            [
                UpConvBlock(in_channels=512, out_channels=256),
                UpConvBlock(in_channels=256, out_channels=128),
                UpConvBlock(in_channels=128, out_channels=64),
            ]
        )
        self.attention_gates = nn.ModuleList(
            [
                AttentionGate(F_g=256, F_l=256, F_int=128),
                AttentionGate(F_g=128, F_l=128, F_int=64),
                AttentionGate(F_g=64, F_l=64, F_int=32),
            ]
        )
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=512,
                    out_channels=256,
                    batchnorm=batchnorm,
                    dropout_rate=dropout_rate,
                ),
                ConvBlock(
                    in_channels=256,
                    out_channels=128,
                    batchnorm=batchnorm,
                    dropout_rate=dropout_rate,
                ),
                ConvBlock(
                    in_channels=128,
                    out_channels=64,
                    batchnorm=batchnorm,
                    dropout_rate=dropout_rate,
                ),
            ]
        )
        self.final_conv = FinalConv(in_channels=64, out_channels=1)

    def forward(self, x, skip_connections):
        for upconv_block, att_gate, conv_block, skip_connection in zip(
            self.upconv_blocks,
            self.attention_gates,
            self.conv_blocks,
            skip_connections[::-1],
        ):
            x = upconv_block(x)
            skip_connection = att_gate(x, skip_connection)
            x = torch.cat((x, skip_connection), dim=1)
            x = conv_block(x)

        x = self.final_conv(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, batchnorm=True, dropout_rate=0.0):
        super(AttentionUNet, self).__init__()
        self.encoder = Encoder(
            in_channels, batchnorm=batchnorm, dropout_rate=dropout_rate
        )
        self.bottleneck = Bottleneck(
            batchnorm=batchnorm, dropout_rate=dropout_rate
        )
        self.decoder = Decoder(
            batchnorm=batchnorm, dropout_rate=dropout_rate
        )

    def forward(self, x):
        output, skip_connections = self.encoder(x)
        output = self.bottleneck(output)
        output = self.decoder(output, skip_connections)
        return output


if __name__ == "__main__":
    model = AttentionUNet(in_channels=3, batchnorm=True, dropout_rate=0.0)
    x = torch.randn((1, 3, 224, 224))
    print(model(x).shape)