"""0.7 x Dice + 0.3 x cross-entropy on every decoder head, plus one term on the summed logits.

This is a simplified version of the paper's multi-head combination loss (which sums the loss
over all 15 combinations of the four heads).
"""
import torch.nn.functional as F


def dice_loss(logits, y, num_classes=4, eps=1e-5):
    p = logits.softmax(1)
    oh = F.one_hot(y, num_classes).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def seg_loss(logits, y):
    return 0.7 * dice_loss(logits, y) + 0.3 * F.cross_entropy(logits, y)


def total_loss(outs, y):
    return sum(seg_loss(o, y) for o in outs) + seg_loss(sum(outs), y)
