
import torch
from typing import Any, Tuple
import torch.nn as nn
import numpy as np

SMOOTH = 1e-8

class DiceLoss(nn.Module):
    def __init__(self, model: nn.Module, l2_weight: float = 1e-8) -> None:
        super(DiceLoss, self).__init__()
        self.model = model
        self.l2_weight = l2_weight

    def forward(self, pred_mask: Any, true_mask: Any, lidar_mask: Any) -> torch.Tensor:

        # include LiDAR mask in computation
        pred_mask = pred_mask * lidar_mask
        true_mask = true_mask * lidar_mask
        
        intersection = torch.sum(pred_mask * true_mask)
        union = torch.sum(pred_mask) + torch.sum(true_mask)

        # Add a small epsilon to the denominator to avoid division by zero
        dice_loss = 1.0 - (2.0 * intersection + SMOOTH) / (union + SMOOTH)

        # Compute L2 regularization loss
        l2_loss = torch.tensor(0.0, requires_grad=True).to(pred_mask.device)
        for param in self.model.parameters():
            l2_loss += torch.norm(param, 2)
        l2_loss = self.l2_weight * l2_loss

        # Combine the dice loss and L2 regularization loss
        total_loss = dice_loss + l2_loss

        return total_loss
    

def calculate_metrics(pred_mask: Any, true_mask: Any, lidar_mask: Any) -> torch.Tensor:
    SMOOTH=1e-8

    true_mask = true_mask.float()
    lidar_mask = lidar_mask.float()

    assert torch.all((pred_mask==0.0) | (pred_mask==1.0)), "pred_mask must be binary"
    assert torch.all((true_mask==0.0) | (true_mask==1.0)), "true_mask must be binary"
    assert torch.all((lidar_mask==0.0) | (lidar_mask==1.0)), "lidar_mask must be binary"

    full_pred_mask = pred_mask
    full_true_mask = true_mask
    
    # include LiDAR mask in computation
    valid_lidar_mask = (lidar_mask == 1.0)
    pred_mask = pred_mask[valid_lidar_mask]
    true_mask = true_mask[valid_lidar_mask]

    intersection = torch.sum(pred_mask * true_mask)
    full_intersection = torch.sum(full_pred_mask * full_true_mask)
    union = torch.sum((pred_mask + true_mask) > 0.5)
    full_union = torch.sum((full_pred_mask + full_true_mask) > 0.5)

    # Add a small epsilon to the denominator to avoid division by zero
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    full_iou = (full_intersection + SMOOTH) / (full_union + SMOOTH)
    dice_score = (2 * intersection + SMOOTH) / (
        torch.sum(pred_mask) + torch.sum(true_mask) + SMOOTH
    )
    pixel_accuracy = torch.sum(pred_mask == true_mask) / true_mask.numel()

    # Calculate TP, FP, TN, FN
    num_TP = torch.sum((pred_mask == 1) & (true_mask == 1))
    num_FP = torch.sum((pred_mask == 1) & (true_mask == 0))
    num_TN = torch.sum((pred_mask == 0) & (true_mask == 0))
    num_FN = torch.sum((pred_mask == 0) & (true_mask == 1))

    assert num_TP + num_FP + num_TN + num_FN == true_mask.numel(), "TP, FP, TN, FN must sum to the total number of pixels"

    precision = num_TP.float() / (num_TP.float() + num_FP.float() + SMOOTH)
    recall = num_TP.float() / (num_TP.float() + num_FN.float() + SMOOTH)
    f1_score = 2 * (precision * recall) / (precision + recall + SMOOTH)

    return iou.item(), full_iou.item(), dice_score.item(), pixel_accuracy.item(), precision.item(), recall.item(), num_TP.item(), num_FP.item(), num_TN.item(), num_FN.item()

def quick_metrics(pred_mask: Any, true_mask: Any, lidar_mask: Any) -> torch.Tensor:
    SMOOTH=1e-8

    true_mask = true_mask.float()
    lidar_mask = lidar_mask.float()

    assert torch.all((pred_mask==0.0) | (pred_mask==1.0)), "pred_mask must be binary"
    assert torch.all((true_mask==0.0) | (true_mask==1.0)), "true_mask must be binary"
    assert torch.all((lidar_mask==0.0) | (lidar_mask==1.0)), "lidar_mask must be binary"

    # include LiDAR mask in computation
    valid_lidar_mask = (lidar_mask == 1.0)
    pred_mask = pred_mask[valid_lidar_mask]
    true_mask = true_mask[valid_lidar_mask]

    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum((pred_mask + true_mask) > 0.5)

    dice_score = (2 * intersection + SMOOTH) / (
        torch.sum(pred_mask) + torch.sum(true_mask) + SMOOTH
    )

    # Add a small epsilon to the denominator to avoid division by zero
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    # Calculate TP, FP, TN, FN
    num_TP = torch.sum((pred_mask == 1) & (true_mask == 1))
    num_FP = torch.sum((pred_mask == 1) & (true_mask == 0))
    num_TN = torch.sum((pred_mask == 0) & (true_mask == 0))
    num_FN = torch.sum((pred_mask == 0) & (true_mask == 1))

    precision = num_TP.float() / (num_TP.float() + num_FP.float() + SMOOTH)
    recall = num_TP.float() / (num_TP.float() + num_FN.float() + SMOOTH)
    f1_score = 2 * (precision * recall) / (precision + recall + SMOOTH)

    return iou.item(), dice_score.item()


def calc_SIC(label, lidar_mask):

    assert torch.all((label==0) | (label==1)), "true_mask must be binary"
    assert torch.all((lidar_mask==0) | (lidar_mask==1)), "lidar_mask must be binary"

    # SIC = # of ice pixels / total # of pixels in LiDAR mask
    img_size = label.size()[1] * label.size()[2]
    label = label[lidar_mask==1]

    # find # of pixels in LiDAR mask
    total_pixels = torch.sum(lidar_mask)

    # find # of ice pixels
    ice_pixels = torch.sum(label)
    sic_lidar = (ice_pixels / total_pixels)

    return sic_lidar.item()

def calc_SIC_rev(label, lidar_mask):
    assert torch.all((label==0) | (label==1)), "label must be binary"
    assert torch.all((lidar_mask==0) | (lidar_mask==1)), "lidar_mask must be binary"

    total_pixels = torch.sum(lidar_mask)

    if total_pixels == 0:
        return float('nan')  # Handle division by zero appropriately

    ice_pixels = torch.sum(label * lidar_mask)
    sic_lidar = ice_pixels / total_pixels

    return sic_lidar.item()

def calc_SIC_np(label, lidar_mask):

    assert np.all((label == 0) | (label == 1)), "label must be binary"
    assert np.all((lidar_mask == 0) | (lidar_mask == 1)), "lidar mask must be binary"

    # SIC = # of ice pixels / total # of pixels in LiDAR mask
    label = label[lidar_mask==1]

    # find # of pixels in LiDAR mask
    total_pixels = np.count_nonzero(lidar_mask==1)

    # find # of ice pixels
    ice_pixels = np.count_nonzero(label==1)
    sic_lidar = (ice_pixels / total_pixels)

    return sic_lidar
    

def calculate_metrics_numpy(pred_mask: Any, true_mask: Any, lidar_mask: Any) -> Tuple[float, float, float, float, float, int, int, int, int]:
    SMOOTH = 1e-8

    assert np.all((pred_mask == 0) | (pred_mask == 1)), "pred_mask must be binary"
    assert np.all((true_mask == 0) | (true_mask == 1)), "true_mask must be binary"
    assert np.all((lidar_mask == 0) | (lidar_mask == 1)), "lidar_mask must be binary"

    # Include LiDAR mask in computation
    valid_lidar_mask = (lidar_mask == 1)
    pred_mask = pred_mask[valid_lidar_mask]
    true_mask = true_mask[valid_lidar_mask]

    pred_mask = pred_mask.astype(np.float32)
    true_mask = true_mask.astype(np.float32)

    intersection = np.sum(pred_mask * true_mask)
    union = np.sum((pred_mask + true_mask) > 0.5)

    # Add a small epsilon to the denominator to avoid division by zero
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice_score = (2 * intersection + SMOOTH) / (np.sum(pred_mask) + np.sum(true_mask) + SMOOTH)
    pixel_accuracy = np.sum(pred_mask == true_mask) / true_mask.size

    # Calculate TP, FP, TN, FN
    num_TP = np.sum((pred_mask == 1) & (true_mask == 1))
    num_FP = np.sum((pred_mask == 1) & (true_mask == 0))
    num_TN = np.sum((pred_mask == 0) & (true_mask == 0))
    num_FN = np.sum((pred_mask == 0) & (true_mask == 1))

    assert num_TP + num_FP + num_TN + num_FN == true_mask.size, "TP, FP, TN, FN must sum to the total number of pixels"

    precision = num_TP / (num_TP + num_FP + SMOOTH)
    recall = num_TP / (num_TP + num_FN + SMOOTH)
    f1_score = 2 * (precision * recall) / (precision + recall + SMOOTH)

    metric_dict = {
        "iou": iou,
        "dice_score": dice_score,
        "pixel_accuracy": pixel_accuracy,
        "precision": precision,
        "recall": recall,
        "num_TP": num_TP,
        "num_FP": num_FP,
        "num_TN": num_TN,
        "num_FN": num_FN,
    }

    return metric_dict


