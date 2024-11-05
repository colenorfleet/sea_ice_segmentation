
import os
import sys
import csv
import torch
import albumentations as A
from utils.dataset_utils import craft_datasetdict, collate_fn, SegmentationDataset
from utils.lossfn_utils import quick_metrics
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import SegformerForSemanticSegmentation
from segmentation_models_pytorch import DeepLabV3Plus, Unet



### define parameters

# "segformer" or "deeplabv3plus" or "unet"
architecture = sys.argv[1]
dataset_name = sys.argv[2]
num_epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])
learning_rate = 5e-3

print(f"Architecture: {architecture}")
print(f"Dataset: {dataset_name}")
print(f"Number of epochs: {num_epochs}")
print(f"Batch size: {batch_size}")


### create dataset

dataset_path = "/home/cole/Documents/NTNU/datasets"

image_dir = os.path.join(dataset_path, "images/")
label_dir = os.path.join(dataset_path, dataset_name, "ice_masks/")
mask_dir = os.path.join(dataset_path, "lidar_masks/")
filename_split_dir = dataset_path

dataset = craft_datasetdict(image_dir, label_dir, mask_dir, filename_split_dir)


ADE_MEAN = [0.4685, 0.4731, 0.4766]
ADE_STD = [0.2034, 0.1987, 0.1968]
img_size = 256


train_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomRotate90(p=0.2),
        A.RandomToneCurve(scale=0.5, p=0.2),
        A.RandomResizedCrop(width=img_size, height=img_size, scale=(0.5,0.75), p=0.2),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ], additional_targets={"lidar_mask": "mask"})

#train_transform = A.Compose([
#        A.Resize(width=img_size, height=img_size),
#        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
#        ], additional_targets={"lidar_mask": "mask"})

val_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ], additional_targets={"lidar_mask": "mask"})

train_dataset = SegmentationDataset(dataset["train"], transform=train_transform)
val_dataset = SegmentationDataset(dataset["val"], transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

### define model

if architecture == 'segformer':
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True,
    )
    model.config.num_labels = 1
    model.config.semantic_loss_ignore_index = -1

elif architecture == 'deeplabv3plus':
    model = DeepLabV3Plus(encoder_name='resnet101', encoder_weights='imagenet', in_channels=3, classes=1)

elif architecture == 'unet':
    model = Unet(encoder_name='resnet101', encoder_weights='imagenet', in_channels=3, classes=1)

###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

## EVALUATE
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, 
        verbose=True, min_lr=1e-6
    )

# Make output directories
output_dir = os.path.join('./output', architecture, dataset_name)
# output_dir = os.path.join('./no_aug_output', architecture, dataset_name)
os.makedirs(output_dir, exist_ok=True)


### training loop
best_val_loss = float("inf")

# Logging
csv_file = os.path.abspath(os.path.join(output_dir, "training_logs.csv"))
csv_header = [
        "Epoch",
        "Avg BCE Train Loss",
        "Avg BCE Val Loss",
        "Learning Rate",
        "Avg Train IOU",
        "Avg Val IOU",
        "Avg Train DICE",
        "Avg Val DICE"
    ]

with open(csv_file, "w+", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_header)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        running_train_loss = 0.0
        running_train_iou = 0.0
        running_train_dice = 0.0
        model.train()

        train_dataloader_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for idx, batch in enumerate(train_dataloader_tqdm):

            image = batch["image"].to(device)
            label = batch["label"].to(device)
            mask = batch["mask"].to(device)

            # forward pass
            if architecture == 'segformer':
                outputs = model(pixel_values=image)
                logits = outputs.logits

                # Upsample
                logits = torch.nn.functional.interpolate(logits, size=(logits.shape[2]*4, logits.shape[3]*4), mode='bilinear', align_corners=False)

            else:
                logits = model(image)

            logits = logits.squeeze(1)

            # Zero the gradients before the backward pass
            optimizer.zero_grad()

            loss = criterion(logits, label.float())
            loss = loss * mask
            epsilon = 1e-8

            mean_loss = loss.sum() / (mask.sum() + epsilon)
            
            mean_loss.backward()
            optimizer.step()

            probabilities = torch.sigmoid(logits)
            pred_mask = (probabilities > 0.5).float()
            iou, dice = quick_metrics(pred_mask, label, mask)

            running_train_loss += mean_loss.item()
            running_train_iou += iou
            running_train_dice += dice

            # update progress bar
            train_dataloader_tqdm.set_postfix(
                loss=mean_loss.item(),
                iou=iou,
                dice=dice
            )

        avg_train_loss = running_train_loss / len(train_dataloader)
        avg_train_iou = running_train_iou / len(train_dataloader)
        avg_train_dice = running_train_dice / len(train_dataloader)

        ### validation loop
        model.eval()
        running_val_loss = 0.0
        running_val_iou = 0.0
        running_val_dice = 0.0

        validation_dataloader_tqdm = tqdm(val_dataloader, desc="Validation", unit="batch")

        with torch.no_grad():
            for idx, batch in enumerate(validation_dataloader_tqdm):

                image = batch["image"].to(device)
                label = batch["label"].to(device)
                mask = batch["mask"].to(device)

                # forward pass
                if architecture == 'segformer':
                    outputs = model(pixel_values=image)
                    logits = outputs.logits
                    # Upsample
                    logits = torch.nn.functional.interpolate(logits, size=(logits.shape[2]*4, logits.shape[3]*4), mode='bilinear', align_corners=False)
                else:
                    logits = model(image)

                logits = logits.squeeze(1)
                v_loss = criterion(logits, label.float())
                v_loss = v_loss * mask

                mean_v_loss = v_loss.sum() / (mask.sum()+ epsilon)
                v_probabilities = torch.sigmoid(logits)
                v_pred_mask = (v_probabilities > 0.5).float()
                val_iou, val_dice = quick_metrics(v_pred_mask, label, mask)

                running_val_loss += mean_v_loss.item()
                running_val_iou += val_iou
                running_val_dice += val_dice

                # update progress bar
                validation_dataloader_tqdm.set_postfix(
                    val_loss=mean_v_loss.item(),
                    val_iou=val_iou,
                    val_f1=val_dice
                )

        avg_val_loss = running_val_loss / len(val_dataloader)
        avg_val_iou = running_val_iou / len(val_dataloader)
        avg_val_dice = running_val_dice / len(val_dataloader)

        scheduler.step(avg_val_loss)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}\n"
            f"Avg Train Loss: {avg_train_loss:.4f}\n"
            f"Avg Validation Loss: {avg_val_loss:.4f}\n"
            f"{'-'*50}"
        )

        if avg_val_loss < best_val_loss:
            print(f"Loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}... saving model")
            torch.save(model.state_dict(), os.path.join(output_dir, "best_ice_seg_model.pth"))
            best_val_loss = avg_val_loss


        # Append the training and validation logs to the CSV file
        current_lr = optimizer.param_groups[0]["lr"]
        csv_writer.writerow(
                [
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    current_lr,
                    avg_train_iou,
                    avg_val_iou,
                    avg_train_dice,
                    avg_val_dice
                ]
            )


    


