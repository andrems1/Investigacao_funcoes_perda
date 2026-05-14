import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoSeg(nn.Module):
    """
    Módulo de segmentação baseado em protótipos.
    Recebe as features profundas (xfeat) e a predição inicial (pred)
    da U-Net e refina a segmentação por similaridade a protótipos.

    Convertido de TensorFlow para PyTorch.
    Referência: Freitas (2025) / Norm_HE_ProtoSeg
    """

    def __init__(self, ndims='2d'):
        super().__init__()

        if ndims == '1d':
            self.dims = (2,)
        elif ndims == '2d':
            self.dims = (2, 3)
        elif ndims == '3d':
            self.dims = (2, 3, 4)
        else:
            raise ValueError('ndims deve ser 1d, 2d ou 3d')

    def forward(self, xfeat, pred, mask=None):
        """
        Args:
            xfeat : features do bottleneck da U-Net  [B, C, H', W']
            pred  : predição inicial do decoder       [B, 1, H,  W ]
            mask  : máscara opcional de tecido        [B, 1, H,  W ]

        Returns:
            pred refinada via protótipos              [B, 2, H,  W ]
        """
        xfeat = xfeat.float()
        pred  = pred.float()

        # ✅ Redimensiona pred para o tamanho espacial do bottleneck (xfeat)
        # xfeat: [B, C, H', W']  ex: [B, 1024, 14, 14]
        # pred:  [B, 1, H,  W ]  ex: [B,    1, 224, 224]
        feat_h, feat_w = xfeat.shape[2], xfeat.shape[3]
        pred_downsampled = F.interpolate(
            pred,
            size=(feat_h, feat_w),
            mode='bilinear',
            align_corners=False,
        )

        if mask is not None:
            mask = mask.float()
            mask_downsampled = F.interpolate(
                mask,
                size=(feat_h, feat_w),
                mode='bilinear',
                align_corners=False,
            )

            pos_prototype = torch.sum(
                xfeat * pred_downsampled * mask_downsampled,
                dim=self.dims, keepdim=True)
            num_pos = torch.sum(
                pred_downsampled * mask_downsampled,
                dim=self.dims, keepdim=True)
            pos_prototype = pos_prototype / (num_pos + 1e-8)

            rpred = 1.0 - pred_downsampled
            neg_prototype = torch.sum(
                xfeat * rpred * mask_downsampled,
                dim=self.dims, keepdim=True)
            num_neg = torch.sum(
                rpred * mask_downsampled,
                dim=self.dims, keepdim=True)
            neg_prototype = neg_prototype / (num_neg + 1e-8)

        else:
            # Protótipos globais (média ponderada sobre todos os pixels)
            pos_prototype = (
                torch.sum(xfeat * pred_downsampled) /
                (torch.sum(pred_downsampled) + 1e-8)
            )
            rpred = 1.0 - pred_downsampled
            neg_prototype = (
                torch.sum(xfeat * rpred) /
                (torch.sum(rpred) + 1e-8)
            )

        # Distância negativa ao protótipo positivo e negativo
        # Calculada no espaço do bottleneck [B, 1, H', W']
        pfeat = -torch.sum((xfeat - pos_prototype) ** 2, dim=1, keepdim=True)
        nfeat = -torch.sum((xfeat - neg_prototype) ** 2, dim=1, keepdim=True)

        # Concatena → [B, 2, H', W']
        disfeat = torch.cat([nfeat, pfeat], dim=1)

        # ✅ Redimensiona de volta para a resolução original da predição [B, 2, H, W]
        pred_h, pred_w = pred.shape[2], pred.shape[3]
        disfeat = F.interpolate(
            disfeat,
            size=(pred_h, pred_w),
            mode='bilinear',
            align_corners=False,
        )

        return F.softmax(disfeat, dim=1)