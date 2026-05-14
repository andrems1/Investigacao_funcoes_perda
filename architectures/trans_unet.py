
import torch.nn as nn
import torch
from einops import rearrange


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
    def __init__(self, in_channels, batchnorm=True, dropout_rate=0.0):
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
                ),
            ]
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            x = self.max_pool(x)
            skip_connections.append(x)
        return x, skip_connections


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1024, patch_size=2, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c", h=H, w=W)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_dim, embed_dim), 
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim=3072, dropout_rate=dropout_rate) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.layers(x)


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


class CascadeUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(CascadeUpSample, self).__init__()
        self.upconv = UpConvBlock(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(FinalLayer, self).__init__()
        self.upconv = UpConvBlock(in_channels, out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.upconv(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, batchnorm=True, dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.cups = nn.ModuleList(
            [
                CascadeUpSample(in_channels=512, out_channels=256, dropout_rate=dropout_rate),
                CascadeUpSample(in_channels=256, out_channels=128, dropout_rate=dropout_rate),
                CascadeUpSample(in_channels=128, out_channels=64, dropout_rate=dropout_rate),
            ]
        )
        self.final_layer = FinalLayer(in_channels=64, out_channels=16, dropout_rate=dropout_rate)

    def forward(self, x, skip_connections):
        for cup, skip_connection in zip(self.cups, skip_connections[::-1]):
            # print(f"Skip connection shape: {skip_connection.shape}")
            # print(f"Input shape: {x.shape}")
            x = cup(x, skip_connection)
            # print(f"Output shape: {x.shape}")
        return self.final_layer(x)
        

class TransUNet(nn.Module):
    def __init__(self, num_heads=12, num_layers=12, patch_size=2, batchnorm=True, dropout_rate=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = Encoder(in_channels=3, batchnorm=batchnorm, dropout_rate=dropout_rate)
        self.patch_embedding = PatchEmbedding(
            in_channels=256, patch_size=patch_size, embed_dim=768
        )
        self.transformer_encoder = TransformerEncoder(
            embed_dim=768, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate
        )
        self.conv1x1 = nn.Conv2d(768, 512, kernel_size=1)
        self.decoder = Decoder(dropout_rate=dropout_rate)
        self.seg_head = SegHead(in_channels=16, out_channels=1)

    def forward(self, x):
        features, skip_connections = self.encoder(x)
        x = self.patch_embedding(features)
        x = self.transformer_encoder(x)
        B, num_patches, embed_dim = x.shape
        H = features.shape[2] // self.patch_size  
        W = features.shape[3] // self.patch_size

        assert H * W == num_patches, f"Inconsistency in the number of patches: {H} x {W} ≠ {num_patches}"
       
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        
        x = self.conv1x1(x)
        x = self.decoder(x, skip_connections)
        return self.seg_head(x)



if __name__ == "__main__":
    model = TransUNet(num_heads=12, num_layers=12, patch_size=2, dropout_rate=0.3)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print("input shape", input.shape)
    print("output shape", output.shape)


