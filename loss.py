import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Para modelos que retornam logits brutos (ex: UNet padrão)."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)   # converte logits → probabilidade
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        return 1 - dice.mean()


class DiceLossProb(nn.Module):
    """Para modelos que já retornam probabilidades (ex: UNetProtoSeg)."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # sem sigmoid — pred já é probabilidade vinda do softmax
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        return 1 - dice.mean()


# Instâncias prontas para uso
bce_logits_loss = nn.BCEWithLogitsLoss()   # espera logits
bce_prob_loss   = nn.BCELoss()             # espera probabilidades [0,1]
dice_loss       = DiceLoss()
dice_loss_prob  = DiceLossProb()


def criterion(pred, target):
    """Função de perda para UNet padrão (saída = logits)."""
    return bce_logits_loss(pred, target) + dice_loss(pred, target)


def criterion_protoseg(pred, target):
    """Função de perda para UNetProtoSeg (saída = probabilidades)."""
    return bce_prob_loss(pred, target) + dice_loss_prob(pred, target)