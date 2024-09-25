
import os
import sys
import csv
import torch
import albumentations as A
from utils.dataset_utils import craft_datasetdict, collate_fn, SegmentationDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import Dinov2Model, SegformerForSemanticSegmentation, SegformerImageProcessor
from utils.dino_utils import DinoBinarySeg
from segmentation_models_pytorch import DeepLabV3Plus, Unet



### define parameters

# "dinov2" or "deeplabv3plus" or "unet"
architecture = sys.argv[1]
dataset_name = sys.argv[2]
num_epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])
learning_rate = 1e-4

print(f"Architecture: {architecture}")
print(f"Dataset: {dataset_name}")
print(f"Number of epochs: {num_epochs}")
print(f"Batch size: {batch_size}")


### create dataset

dataset_path = "/home/cole/Documents/NTNU/datasets"

image_dir = os.path.join(dataset_path, dataset_name, "images/")
label_dir = os.path.join(dataset_path, dataset_name, "ice_masks/")
mask_dir = os.path.join(dataset_path, dataset_name, "lidar_masks/")
filename_split_dir = dataset_path

dataset = craft_datasetdict(image_dir, label_dir, mask_dir, filename_split_dir)


### NEED TO CHECK: are all input images in the range [0,1]?
# Can use AutoImageProcessor to see (from transformers)

ADE_MEAN = [0.4685, 0.4731, 0.4766]
ADE_STD = [0.2034, 0.1987, 0.1968]
img_size = 512

# A.HorizontalFlip(p=0.5),
# A.Normalize(mean=ADE_MEAN, std=ADE_STD)],
train_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], additional_targets={"lidar_mask": "mask"})

val_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], additional_targets={"lidar_mask": "mask"})

train_dataset = SegmentationDataset(dataset["train"], transform=train_transform)
val_dataset = SegmentationDataset(dataset["val"], transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

### define model
'''
if architecture == 'dinov2':
    
    dinov2_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
    deeplabv3plus_decoder = DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1)

    for param in dinov2_encoder.parameters():
        param.requires_grad = False
    
    model = DinoBinarySeg(encoder=dinov2_encoder, decoder=deeplabv3plus_decoder)
'''
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

    model = Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## EVALUATE
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

model.to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
optimizer = AdamW(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#        optimizer, mode="min", factor=0.1, patience=3, verbose=True
#    )

# Make output directories
output_dir = os.path.join('./output', architecture, dataset_name)
os.makedirs(output_dir, exist_ok=True)


### training loop
best_val_loss = float("inf")
model.train()

# Logging
csv_file = os.path.abspath(os.path.join(output_dir, "training_logs.csv"))
csv_header = [
        "Epoch",
        "Avg BCE Train Loss",
        "Avg BCE Val Loss",
        "Learning Rate",
    ]

with open(csv_file, "w+", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_header)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        running_train_loss = 0.0

        train_dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/ {num_epochs}", unit="batch")

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

            running_train_loss += mean_loss.item()

            # update progress bar
            train_dataloader.set_postfix(
                loss=mean_loss.item()
            )

        avg_train_loss = running_train_loss / len(train_dataloader)

        ### validation loop
        model.eval()
        running_val_loss = 0.0

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
                running_val_loss += mean_v_loss.item()

                # update progress bar
                validation_dataloader.set_postfix(
                    val_loss=mean_v_loss.item()
                )

        avg_val_loss = running_val_loss / len(validation_dataloader)

        # scheduler.step(avg_val_loss)

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
        csv_writer.writerow(
                [
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    learning_rate,
                ]
            )


    


