
import os
import sys
import csv
import torch
from utils.dataset_utils import craft_labelled_dataset, collate_fn, SegmentationDataset
import albumentations as A
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation
from utils.lossfn_utils import calculate_metrics, calc_SIC
from utils.plotting_utils import save_segmentation_image
from segmentation_models_pytorch import DeepLabV3Plus, Unet

architecture = sys.argv[1]
eval_dataset_name = sys.argv[2] # should be GoNorth or roboflow
# train_dataset_name = sys.argv[3] # should be raw, morph, otsu 

print(f"Architecture: {architecture}")
print(f"Eval Dataset: {eval_dataset_name}")
# print(f"Train Dataset: {train_dataset_name}")

dataset_path = "/home/cole/Documents/NTNU/datasets/labelled"
image_dir = os.path.join(dataset_path, eval_dataset_name, "images/")
label_dir = os.path.join(dataset_path, eval_dataset_name, "ice_masks/")
mask_dir = os.path.join(dataset_path, eval_dataset_name, "lidar_masks/")

dataset = craft_labelled_dataset(image_dir, label_dir, mask_dir)

ADE_MEAN = [0.4685, 0.4731, 0.4766]
ADE_STD = [0.2034, 0.1987, 0.1968]
img_size = 512

val_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)],
        additional_targets={"lidar_mask": "mask"})

test_dataset = SegmentationDataset(dataset, transform=val_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

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

# data_directory = os.path.join('./output', architecture, train_dataset_name)
data_directory = os.path.join('./all_dataset_output', architecture)

model.load_state_dict(torch.load(os.path.join(data_directory, "best_ice_seg_model.pth")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

# test_data_output_dir = os.path.join("./all_dataset_labelled_output", architecture, eval_dataset_name, train_dataset_name)
test_data_output_dir = os.path.join("./all_dataset_labelled_output", architecture, eval_dataset_name)
os.makedirs(test_data_output_dir, exist_ok=True)

# Logging
csv_file = os.path.abspath(os.path.join(test_data_output_dir, "evaluation_scores.csv"))
csv_header = [
    "Sample",
    "BCE Loss",
    "Total BCE Loss",
    "IOU",
    "DICE",
    "Pixel Accuracy",
    "Precision",
    "Recall",
    "Number True Positive",
    "Number False Positive",
    "Number True Negative",
    "Number False Negative",
    "SIC Label",
    "SIC Pred",
]

model.eval()

with open(csv_file, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_header)

    with torch.no_grad():
        
        test_dataloader = tqdm(test_dataloader, desc="Evaluation", unit="image")

        for idx, batch in enumerate(test_dataloader):

            image = batch["image"].to(device)
            label = batch["label"].to(device)
            mask = batch["mask"].to(device)
            filename = batch["filename"]
            
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
            unmasked_mean_loss = loss.sum() / (img_size * img_size)
            loss = loss * mask
            mean_loss = loss.sum() / mask.sum()

            # calculate metrics
            pred_mask = torch.where(logits > 0.5, 1, 0)
            iou, dice, pixel_accuracy, precision, recall, num_TP, num_FP, num_TN, num_FN = calculate_metrics(pred_mask, label, mask)
            sic_label = calc_SIC(label, mask)
            sic_pred = calc_SIC(pred_mask, mask)

            csv_writer.writerow(
                [filename[0],
                 mean_loss.item(),
                 unmasked_mean_loss.item(),
                 iou,
                 dice,
                 pixel_accuracy,
                 precision,
                 recall,
                 num_TP,
                 num_FP,
                 num_TN,
                 num_FN,
                 sic_label,
                 sic_pred]
                 )
            
            # save image
            save_segmentation_image(batch['original_image'], label, pred_mask, filename, test_data_output_dir)



