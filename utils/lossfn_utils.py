
import torch
from typing import Any
import torch.nn as nn

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

    assert torch.all((pred_mask==0) | (pred_mask==1)), "pred_mask must be binary"
    assert torch.all((true_mask==0) | (true_mask==1)), "true_mask must be binary"
    assert torch.all((lidar_mask==0) | (lidar_mask==1)), "lidar_mask must be binary"
    
    # include LiDAR mask in computation
    valid_lidar_mask = (lidar_mask == 1)
    pred_mask = pred_mask[valid_lidar_mask]
    true_mask = true_mask[valid_lidar_mask]

    pred_mask = pred_mask.float()
    true_mask = true_mask.float()

    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum((pred_mask + true_mask) > 0.5)

    # Add a small epsilon to the denominator to avoid division by zero
    iou = (intersection + SMOOTH) / (union + SMOOTH)
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

    return iou.item(), dice_score.item(), pixel_accuracy.item(), precision.item(), recall.item(), num_TP.item(), num_FP.item(), num_TN.item(), num_FN.item()

def quick_metrics(pred_mask: Any, true_mask: Any, lidar_mask: Any) -> torch.Tensor:

    pred_mask = torch.where(pred_mask > 0.5, 1, 0)

    assert torch.all((pred_mask==0) | (pred_mask==1)), "pred_mask must be binary"
    assert torch.all((true_mask==0) | (true_mask==1)), "true_mask must be binary"
    assert torch.all((lidar_mask==0) | (lidar_mask==1)), "lidar_mask must be binary"

    # include LiDAR mask in computation
    valid_lidar_mask = (lidar_mask == 1)
    pred_mask = pred_mask[valid_lidar_mask]
    true_mask = true_mask[valid_lidar_mask]
    
    pred_mask = pred_mask.float()
    true_mask = true_mask.float()

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

    # SIC = # of ice pixels / total # of pixels in LiDAR mask
    img_size = label.size()[1] * label.size()[2]
    label = label[lidar_mask==1]

    # find # of pixels in LiDAR mask
    total_pixels = torch.sum(lidar_mask)

    # find # of ice pixels
    ice_pixels = torch.sum(label)
    sic_lidar = (ice_pixels / total_pixels)

    return sic_lidar.item()

