
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
    

def calculate_metrics(pred_mask: Any, true_mask: Any, lidar_mask: Any) -> torch.Tensor:#, mask: Any) -> torch.Tensor:

    
    # include LiDAR mask in computation
    pred_mask = pred_mask * lidar_mask
    true_mask = true_mask * lidar_mask
    
    pred_mask = pred_mask.float()
    true_mask = true_mask.float()

    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum((pred_mask + true_mask) > 0.5)

    # Add a small epsilon to the denominator to avoid division by zero
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice_coefficient = (2 * intersection + SMOOTH) / (
        torch.sum(pred_mask) + torch.sum(true_mask) + SMOOTH
    )
    pixel_accuracy = torch.sum(pred_mask == true_mask) / true_mask.numel()

    # Accuracy for foreground (TP / Total Positives in true mask)
    foreground_mask = (true_mask == 1)
    foreground_accuracy = torch.sum(pred_mask[foreground_mask] == true_mask[foreground_mask]) / torch.sum(foreground_mask)

    # Accuracy for background (TN / Total Negatives in true mask)
    background_mask = (true_mask == 0)
    background_accuracy = torch.sum(pred_mask[background_mask] == true_mask[background_mask]) / torch.sum(background_mask)

    # False Negative Rate (FN / Total Positives in true mask)
    false_negatives = torch.sum((pred_mask == 0) & (true_mask == 1))
    false_positive_rate = false_negatives / torch.sum(foreground_mask)

    # Calculate TP, FP, TN, FN
    num_TP = torch.sum((pred_mask == 1) & (true_mask == 1))
    num_FP = torch.sum((pred_mask == 1) & (true_mask == 0))
    num_TN = torch.sum((pred_mask == 0) & (true_mask == 0))
    num_FN = torch.sum((pred_mask == 0) & (true_mask == 1))

    precision = num_TP.float() / (num_TP.float() + num_FP.float() + SMOOTH)
    recall = num_TP.float() / (num_TP.float() + num_FN.float() + SMOOTH)
    f1_score = 2 * (precision * recall) / (precision + recall + SMOOTH)

    return iou.item(), dice_coefficient.item(), pixel_accuracy.item(), foreground_accuracy.item(), background_accuracy.item(), false_positive_rate.item(), num_TP.item(), num_FP.item(), num_TN.item(), num_FN.item(), f1_score.item()



def calc_SIC(label, lidar_mask):

    
    # count number of pixels = 1 and = 0
    # apply LiDAR mask
    # SIC = # of ice pixels / total # of pixels in LiDAR mask

    img_size = label.size()[1] * label.size()[2]

    label = label * lidar_mask

    # find # of pixels in LiDAR mask
    total_pixels = torch.sum(lidar_mask)

    # find # of ice pixels
    ice_pixels = torch.sum(label)
    
    sic_total = (ice_pixels / img_size)
    sic_lidar = (ice_pixels / total_pixels)

    return sic_lidar.item()