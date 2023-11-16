import numpy as np
import torch

def confusion_matrix(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def get_miou(conf_matrix):
    inter = torch.diag(conf_matrix)

    true = conf_matrix.sum(dim=1)
    pred = conf_matrix.sum(dim=0)

    union = true + pred - inter

    iou_per_class = inter / (union + 1e-6)
    miou = torch.mean(iou_per_class)
    return miou, iou_per_class

def test_iou(gt, pred):
  gt_mask = torch.zeros(10, 10).long()
  gt_mask[gt[0]:gt[0] + gt[2], gt[1]:gt[1] + gt[3]] = 1
  pred_mask = torch.zeros(10, 10).long()
  pred_mask[pred[0]:pred[0] + pred[2], pred[1]:pred[1] + pred[3]] = 1

  fig, ax = plt.subplots(figsize=(6, 6))
  ax.set_ylim(0, 5)
  ax.set_xlim(0, 5)
  ax.add_patch(plt.Rectangle((gt[0], gt[1]), gt[2], gt[3], color='green', alpha=0.5));
  ax.add_patch(plt.Rectangle((pred[0], pred[1]), pred[2], pred[3], color='red', alpha=0.5));
  iou = get_miou(confusion_matrix(gt_mask, pred_mask, 2))[1][1].item()
  ax.set_title(f"IoU = {round(iou, 2)}")