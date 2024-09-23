
import os
import sys
import csv
import torch
from dataset_utils import craft_datasetdict, collate_fn, SegmentationDataset
from dino_utils import SegmentationHead, DinoBinarySeg
import albumentations as A
from torch.utils.data import DataLoader
from model.unet_brain_seg import UNet as UNetBrainSeg
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
from tqdm.auto import tqdm
from transformers import Dinov2Model
from lossfn_utils import calculate_metrics, calc_SIC
from plotting_utils import save_segmentation_image
from segmentation_models_pytorch import PSPNet, DeepLabV3Plus, Unet, DeepLabV3


architecture = sys.argv[1]
dataset_name = sys.argv[2]

print(f"Architecture: {architecture}")
print(f"Dataset: {dataset_name}")

### create dataset

dataset_path = "/home/cole/Documents/NTNU/datasets"

image_dir = os.path.join(dataset_path, dataset_name, "images/")
label_dir = os.path.join(dataset_path, dataset_name, "ice_masks/")
mask_dir = os.path.join(dataset_path, dataset_name, "lidar_masks/")
filename_split_dir = dataset_path

dataset = craft_datasetdict(image_dir, label_dir, mask_dir, filename_split_dir)


ADE_MEAN = [0.4684301, 0.47295512, 0.47658848]
ADE_STD = [0.20301826, 0.19884902, 0.1973144]
img_size = 448


val_transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)],
        additional_targets={"lidar_mask": "mask"})


test_dataset = SegmentationDataset(dataset["test"], transform=val_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

data_directory = os.path.join('./output', architecture, dataset_name)


if architecture == 'dinov2':
    ### DINOV2
    # create id2label for dinov2
    dinov2_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
    deeplab_decoder = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    deeplab_decoder.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    deeplab_decoder.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    # dinov2_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    for param in dinov2_encoder.parameters():
        param.requires_grad = False
    # dinov2_encoder.load_state_dict(torch.load('dinov2_vitb14_voc2012_linear_head.pth'))

    # segmentation_head = SegmentationHead(in_channels=768, num_classes=1)
    model = DinoBinarySeg(encoder=dinov2_encoder, decoder=deeplab_decoder.classifier)

elif architecture == 'deeplabv3':
    ### DEEPLABV3
    model = DeepLabV3(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    #model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    #model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
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


model.load_state_dict(torch.load(os.path.join(data_directory, "best_ice_seg_model.pth")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()

test_data_output_dir = os.path.join("./test_data_output", architecture, dataset_name)
os.makedirs(test_data_output_dir, exist_ok=True)

# Logging
csv_file = os.path.abspath(os.path.join(test_data_output_dir, "evaluation_scores.csv"))
csv_header = [
    "Sample",
    "BCE Loss",
    "IOU",
    "Dice Coefficient",
    "Pixel Accuracy",
    "Foreground Accuracy",
    "Background Accuracy",
    "False Negative Rate",
    "SIC Label",
    "SIC Prediction",
    "Number True Positive",
    "Number False Positive",
    "Number True Negative",
    "Number False Negative",
    "F1 Score",
]

model.eval()



with open(csv_file, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_header)

    with torch.no_grad():
        
        train_dataloader = tqdm(test_dataloader, desc="Evaluation", unit="image")

        for idx, batch in enumerate(train_dataloader):

            image = batch["image"].to(device)
            label = batch["label"].to(device)
            mask = batch["mask"].to(device)
            filename = batch["filename"]


            # forward pass
            output = model(image)

            #if architecture == 'deeplabv3':
            #    output = output["out"]

            output = output.squeeze() * mask
            label = label * mask

            t_loss = criterion(output, label.float()) #, mask)

            # calculate metrics

            pred_mask = torch.where(output > 0.5, 1, 0)
            iou, dice_coefficient, pixel_accuracy, foreground_accuracy, background_accuracy, false_negative_rate, num_TP, num_FP, num_TN, num_FN, f1_score = calculate_metrics(pred_mask, label, mask)
            sic_label = calc_SIC(label, mask)
            sic_pred = calc_SIC(pred_mask, mask)

            csv_writer.writerow(
                [filename[0],
                 t_loss.item(),
                 iou,
                 dice_coefficient,
                 pixel_accuracy,
                 foreground_accuracy,
                 background_accuracy,
                 false_negative_rate,
                 sic_label,
                 sic_pred,
                 num_TP,
                 num_FP,
                 num_TN,
                 num_FN,
                 f1_score]
                 )
            
            # save image
            save_segmentation_image(batch['original_image'], label, pred_mask, filename, test_data_output_dir)



