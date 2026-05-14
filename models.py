from architectures import UNet, UNetPP, SharpUNet, AttentionUNet, ResidualUNet, TransUNet
from architectures.unet import UNetProtoSeg  # <- adiciona esse import

def build_model(experiment_model, dropout_rate, batchnorm, device):
    if experiment_model == "UNet":
        model = UNet(in_channels=3, batchnorm=batchnorm, dropout_rate=dropout_rate)
    elif experiment_model == "UNetProtoSeg":  # <- adiciona esse bloco
        model = UNetProtoSeg(in_channels=3, batchnorm=batchnorm, dropout_rate=dropout_rate)
    elif experiment_model == "UNetPP":
        model = UNetPP(in_channels=3, batchnorm=batchnorm, dropout_rate=dropout_rate)
    elif experiment_model == "SharpUNet":
        model = SharpUNet(in_channels=3, batchnorm=batchnorm, dropout_rate=dropout_rate)
    elif experiment_model == "AttentionUNet":
        model = AttentionUNet(in_channels=3, batchnorm=batchnorm, dropout_rate=dropout_rate)
    elif experiment_model == "ResidualUNet":
        model = ResidualUNet(in_channels=3, dropout_rate=dropout_rate)
    elif experiment_model == "TransUNet":
        model = TransUNet(dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model: {experiment_model}")
    return model.to(device)