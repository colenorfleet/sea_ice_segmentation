import os
import sys
import csv
import torch
import albumentations as A
from utils.dataset_utils import craft_cv_datasetdict, collate_fn, SegmentationDataset
from utils.lossfn_utils import quick_metrics
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import SegformerForSemanticSegmentation
from segmentation_models_pytorch import DeepLabV3Plus, Unet
from sklearn.model_selection import KFold



# "segformer" or "deeplabv3plus" or "unet"
architecture = sys.argv[1]
dataset_name = sys.argv[2]
num_epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])
learning_rate = 1e-3
num_folds = 5

print(f"Cross-Validation Training with {num_folds} folds")
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

cv_dataset = craft_cv_datasetdict(image_dir, label_dir, mask_dir, filename_split_dir)

ADE_MEAN = [0.4685, 0.4731, 0.4766]
ADE_STD = [0.2034, 0.1987, 0.1968]
img_size = 512

train_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomResizedCrop(width=img_size, height=img_size, scale=(0.5,0.75), p=0.2),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ], additional_targets={"lidar_mask": "mask"})

val_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ], additional_targets={"lidar_mask": "mask"})


kfold = KFold(n_splits=num_folds, shuffle=True)

# Make output directories
output_dir = os.path.join('./cv_output', architecture, dataset_name)
os.makedirs(output_dir, exist_ok=True)


### Cross-Validation Training Loop ###

for fold, (train_indices, val_indices) in enumerate(kfold.split(cv_dataset['cv_train'])):
    print(f"Fold {fold}")

    train_subset = Subset(cv_dataset['cv_train'], [int(idx) for idx in train_indices])
    val_subset = Subset(cv_dataset['cv_train'], [int(idx) for idx in val_indices])

    train_subset_cv = SegmentationDataset(train_subset, train_transform)
    val_subset_cv = SegmentationDataset(val_subset, val_transform)

    train_dataloader = DataLoader(train_subset_cv, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_subset_cv, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    # initialize model here, new one everytime
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
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    ### training loop ###
    best_val_loss = float("inf")
    model.train()

    # Logging
    csv_file = os.path.abspath(os.path.join(output_dir, f"training_logs_fold_{fold}.csv"))
    csv_header = [
            "Epoch",
            "Avg BCE Train Loss",
            "Avg BCE Val Loss",
            "Learning Rate",
            "Avg Train IOU",
            "Avg Val IOU",
            "Avg Train F1",
            "Avg Val F1"
        ]
    
    with open(csv_file, "w+", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            running_train_loss = 0.0
            running_train_iou = 0.0
            running_train_f1 = 0.0

            train_dataloader = tqdm(train_dataloader, desc=f"Fold {fold} Epoch {epoch + 1}/ {num_epochs}", unit="batch")

            for idx, batch in enumerate(train_dataloader):

                image = batch["image"].to(device)
                label = batch["label"].to(device)
                mask = batch["mask"].to(device)

                # forward pass
                if architecture == 'segformer':
                    outputs = model(pixel_values=image)
                    logits = outputs.logits

                    # Upsample
                    logits = torch.nn.functional.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

                else:
                    logits = model(image)

                logits = logits.squeeze(1)
                loss = criterion(logits, label.float())
                loss = loss * mask

                mean_loss = loss.sum() / mask.sum()
                    
                mean_loss.backward()
                optimizer.step()

                optimizer.zero_grad()

                iou, f1 = quick_metrics(logits, label, mask)

                running_train_loss += mean_loss.item()
                running_train_iou += iou
                running_train_f1 += f1

                # update progress bar
                train_dataloader.set_postfix(
                    loss=mean_loss.item(),
                    iou=iou,
                    f1=f1
                )

            avg_train_loss = running_train_loss / len(train_dataloader)
            avg_train_iou = running_train_iou / len(train_dataloader)
            avg_train_f1 = running_train_f1 / len(train_dataloader)



            ### validation loop ###
            model.eval()
            running_val_loss = 0.0
            running_val_iou = 0.0
            running_val_f1 = 0.0

            validation_dataloader = tqdm(val_dataloader, desc="Validation", unit="batch")

            with torch.no_grad():
                for idx, batch in enumerate(validation_dataloader):

                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    mask = batch["mask"].to(device)

                    # forward pass
                    if architecture == 'segformer':
                        outputs = model(pixel_values=image)
                        logits = outputs.logits
                        # Upsample
                        logits = torch.nn.functional.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
                    else:
                        logits = model(image)

                    logits = logits.squeeze(1)    
                    v_loss = criterion(logits, label.float())
                    v_loss = v_loss * mask

                    mean_v_loss = v_loss.sum() / mask.sum()
                    val_iou, val_f1 = quick_metrics(logits, label, mask)

                    running_val_loss += mean_v_loss.item()
                    running_val_iou += val_iou
                    running_val_f1 += val_f1

                    # update progress bar
                    validation_dataloader.set_postfix(
                        val_loss=mean_v_loss.item(),
                        val_iou=val_iou,
                        val_f1=val_f1
                    )

            avg_val_loss = running_val_loss / len(validation_dataloader)
            avg_val_iou = running_val_iou / len(validation_dataloader)
            avg_val_f1 = running_val_f1 / len(validation_dataloader)

            scheduler.step(avg_val_loss)

            print(
                f"\nFold {fold}\n"
                f"Epoch {epoch + 1}/{num_epochs}\n"
                f"Avg Train Loss: {avg_train_loss:.4f}\n"
                f"Avg Validation Loss: {avg_val_loss:.4f}\n"
                f"{'-'*50}"
            )

            if avg_val_loss < best_val_loss:
                print(f"Loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}... saving model")
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_ice_seg_model_fold_{fold}.pth"))
                best_val_loss = avg_val_loss


            # Append the training and validation logs to the CSV file
            csv_writer.writerow(
                    [
                        epoch + 1,
                        avg_train_loss,
                        avg_val_loss,
                        learning_rate,
                        avg_train_iou,
                        avg_val_iou,
                        avg_train_f1,
                        avg_val_f1
                    ]
                )


