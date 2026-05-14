import torch

def iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-7) / (union + 1e-7)


def dice_coeff(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)


def precision_recall(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return precision, recall


def specificity(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return tn / (tn + fp + 1e-7)


def accuracy(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    correct = (pred == target).sum()
    total = target.numel()
    return correct / total