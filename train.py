
import os
import sys
import csv
import torch
import albumentations as A
from dataset_utils import craft_datasetdict, collate_fn, SegmentationDataset, Dinov2forSemanticSegmentation
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
# import evaluate
import numpy as np
from model.unet_brain_seg import UNet as UNetBrainSeg
from lossfn_utils import DiceLoss
from torchvision.models.segmentation import deeplabv3_resnet50
from transformers import Dinov2Model
from dino_utils import SegmentationHead, DinoBinarySeg
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

from segmentation_models_pytorch import PSPNet, DeepLabV3Plus, Unet, DeepLabV3



### define parameters

# "dinov2" or "deeplabv3" or "unet"
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

ADE_MEAN = [0.4684301, 0.47295512, 0.47658848]
ADE_STD = [0.20301826, 0.19884902, 0.1973144]
img_size = 448


train_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)],
        additional_targets={"lidar_mask": "mask"})

val_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)],
        additional_targets={"lidar_mask": "mask"})

train_dataset = SegmentationDataset(dataset["train"], transform=train_transform)
val_dataset = SegmentationDataset(dataset["val"], transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

### define model

if architecture == 'dinov2':
    
    dinov2_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
    deeplabv3plus_decoder = DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1)

    # deeplab_decoder = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # deeplab_decoder.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # deeplab_decoder.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    # dinov2_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    for param in dinov2_encoder.parameters():
        param.requires_grad = False
    # dinov2_encoder.load_state_dict(torch.load('dinov2_vitb14_voc2012_linear_head.pth'))

    # segmentation_head = SegmentationHead(in_channels=768, num_classes=1)
    model = DinoBinarySeg(encoder=dinov2_encoder, decoder=deeplabv3plus_decoder)

elif architecture == 'deeplabv3':
    ### DEEPLABV3

    model = DeepLabV3(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    # model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # model.load_state_dict(torch.load('best_deeplabv3_resnet50.pth'))

elif architecture == 'unet_brain':
    ### UNET
    model = UNetBrainSeg(in_channels=3, out_channels=1, init_features=32)
    model.load_state_dict(torch.load('./pretrained/brain_seg_pretrained.pth'))

elif architecture == 'pspnet':

    model = PSPNet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1)

elif architecture == 'deeplabv3plus':

    model = DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1)

elif architecture == 'unet_smp':

    model = Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## EVALUATE
# train_metrics = evaluate.combine(["mean_iou", "accuracy"])
# eval_metrics = evaluate.combine(["mean_iou", "accuracy"])

# criterion = DiceLoss(model=model)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()

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
            output = model(image)
          
            #if architecture == 'deeplabv3':
            #    output = output["out"]


            # calculate loss
            # train_metrics.add_batch(predictions=output, references=label)

            output = output.squeeze() * mask
            label = label * mask

            t_loss = criterion(output, label.float()) #, mask)
            t_loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            running_train_loss += t_loss.item()

            # update progress bar
            train_dataloader.set_postfix(
                loss=t_loss.item()
            )

        avg_train_loss = running_train_loss / len(train_dataloader)

        # train_met = train_metrics.compute(num_labels=1)
        # print(f"Train Loss: {train_met['mean_iou']:.4f}")


        ### validation loop
        model.eval()
        running_val_loss = 0.0

        validation_dataloader = tqdm(val_dataloader, desc="Validation", unit="batch")

        with torch.no_grad():
            for idx, batch in enumerate(validation_dataloader):

                image = batch["image"].to(device)
                label = batch["label"].to(device)
                mask = batch["mask"].to(device)

                output = model(image)

                #if architecture == 'deeplabv3':
                #    output = output["out"]

                # eval_metrics.add_batch(predictions=output, references=label)

                output = output.squeeze() * mask
                label = label * mask

                v_loss = criterion(output, label.float())#, mask)
                running_val_loss += v_loss.item()

                # update progress bar
                validation_dataloader.set_postfix(
                    val_loss=v_loss.item()
                )

        # eval_met = eval_metrics.compute(num_labels=1)
        # print(f"Validation Loss: {eval_met['mean_iou']:.4f}")
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


    


