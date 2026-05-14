import torch
import torch.nn as nn
from protoseg import ProtoSeg


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, batchnorm=True, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, batchnorm, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_channels=in_channels, out_channels=64,
                      batchnorm=batchnorm, dropout_rate=dropout_rate),
            ConvBlock(in_channels=64, out_channels=128,
                      batchnorm=batchnorm, dropout_rate=dropout_rate),
            ConvBlock(in_channels=128, out_channels=256,
                      batchnorm=batchnorm, dropout_rate=dropout_rate),
            ConvBlock(in_channels=256, out_channels=512,
                      batchnorm=batchnorm, dropout_rate=dropout_rate),
        ])
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
        self.conv = ConvBlock(in_channels=512, out_channels=1024,
                              batchnorm=batchnorm, dropout_rate=dropout_rate)

    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size, stride=stride)

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
        self.upconv_blocks = nn.ModuleList([
            UpConvBlock(in_channels=1024, out_channels=512),
            UpConvBlock(in_channels=512,  out_channels=256),
            UpConvBlock(in_channels=256,  out_channels=128),
            UpConvBlock(in_channels=128,  out_channels=64),
        ])
        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_channels=1024, out_channels=512,
                      batchnorm=batchnorm, dropout_rate=dropout_rate),
            ConvBlock(in_channels=512,  out_channels=256,
                      batchnorm=batchnorm, dropout_rate=dropout_rate),
            ConvBlock(in_channels=256,  out_channels=128,
                      batchnorm=batchnorm, dropout_rate=dropout_rate),
            ConvBlock(in_channels=128,  out_channels=64,
                      batchnorm=batchnorm, dropout_rate=dropout_rate),
        ])
        self.final_conv = FinalConv(in_channels=64, out_channels=1)

    def forward(self, x, skip_connections):
        for upconv_block, conv_block, skip_connection in zip(
            self.upconv_blocks, self.conv_blocks, skip_connections[::-1]
        ):
            x = upconv_block(x)
            x = torch.cat((x, skip_connection), dim=1)
            x = conv_block(x)
        x = self.final_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.0, batchnorm=True):
        super(UNet, self).__init__()
        self.encoder    = Encoder(in_channels, batchnorm, dropout_rate=dropout_rate)
        self.bottleneck = Bottleneck(batchnorm, dropout_rate=dropout_rate)
        self.decoder    = Decoder(batchnorm, dropout_rate=dropout_rate)

    def forward(self, x):
        output, skip_connections = self.encoder(x)
        output = self.bottleneck(output)
        output = self.decoder(output, skip_connections)
        return output

    def forward_with_bottleneck(self, x):
        """
        Igual ao forward normal, mas também retorna
        as features do bottleneck para o ProtoSeg.
        """
        # ✅ Nomes corretos: self.encoder, self.bottleneck, self.decoder
        encoder_out, skip_connections = self.encoder(x)
        bottleneck_feat = self.bottleneck(encoder_out)
        pred = self.decoder(bottleneck_feat, skip_connections)
        return bottleneck_feat, pred


class UNetProtoSeg(nn.Module):
    """
    U-Net com módulo ProtoSeg acoplado ao bottleneck.
    A predição inicial do decoder é refinada pelos protótipos
    calculados sobre as features do bottleneck.
    """

    def __init__(self, in_channels=3, dropout_rate=0.2, batchnorm=True):
        super().__init__()
        # ✅ in_channels adicionado (estava faltando)
        self.unet     = UNet(in_channels=in_channels, dropout_rate=dropout_rate,
                             batchnorm=batchnorm)
        self.protoseg = ProtoSeg(ndims='2d')

    def forward(self, x):
        # 1) Features do bottleneck + predição inicial da U-Net
        bottleneck_feat, initial_pred = self.unet.forward_with_bottleneck(x)

        # 2) Converte para probabilidade [0,1]
        pred_prob = torch.sigmoid(initial_pred)

        # 3) Refina com ProtoSeg → [B, 2, H, W]
        refined = self.protoseg(bottleneck_feat, pred_prob)

        # 4) Retorna só o canal positivo (núcleo) → [B, 1, H, W]
        return refined[:, 1:2, :, :]


if __name__ == "__main__":
    model = UNet(in_channels=3, batchnorm=False)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print("input shape:", input.shape)
    print("output shape:", output.shape)

    model_proto = UNetProtoSeg(in_channels=3)
    output_proto = model_proto(input)
    print("UNetProtoSeg output shape:", output_proto.shape)