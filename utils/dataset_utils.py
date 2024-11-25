
from datasets import Dataset, DatasetDict, Image
import torch
import os
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput



def create_dataset(image_paths, label_paths, mask_paths, filenames):
    dataset = Dataset.from_dict({"image": sorted(image_paths), "label": sorted(label_paths), "mask": sorted(mask_paths), "filename": sorted(filenames)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())
    dataset = dataset.cast_column("mask", Image())

    return dataset

class SegmentationDataset(TorchDataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        original_image = np.array(item["image"])
        original_segmentation_map = np.array(item["label"]) / 255
        original_mask = np.array(item["mask"]) / 255
        filename = item["filename"]
        
        transformed = self.transform(image=original_image,
                                     mask=original_segmentation_map,
                                     lidar_mask=original_mask)
        
        image = torch.tensor(transformed["image"])
        label = torch.LongTensor(transformed["mask"])
        mask = torch.LongTensor(transformed["lidar_mask"])

        assert torch.all((label==0) | (label==1)), "label is not binary"
        assert torch.all((mask==0) | (mask==1)), "mask is not binary"

        image = image.permute(2, 0, 1)

        return image, label, mask, original_image, original_segmentation_map, filename

def collate_fn(inputs):
    batch = dict()
    batch["image"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["label"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["mask"] = torch.stack([i[2] for i in inputs], dim=0)
    batch["original_image"] = [i[3] for i in inputs]
    batch["original_segmentation_map"] = [i[4] for i in inputs]
    batch["filename"] = [i[5] for i in inputs]

    return batch

def read_split_files(img_path_dir, label_path_dir, mask_path_dir, split_path, split_names):
    image_paths, label_paths, mask_paths, filenames = {}, {}, {}, {}
    for split in split_names:
        image_paths[split], label_paths[split], mask_paths[split], filenames[split] = [],[],[],[]
        with open(os.path.join(split_path, f"{split}.txt"), "r") as f:
            for line in f:
                filename = line.strip()
                image_paths[split].append(os.path.join(img_path_dir, f"{filename}.jpg"))
                label_paths[split].append(os.path.join(label_path_dir, f"{filename}.png"))
                mask_paths[split].append(os.path.join(mask_path_dir, f"{filename}.png"))
                filenames[split].append(filename)
    return image_paths, label_paths, mask_paths, filenames

def craft_datasetdict(img_path_dir, label_path_dir, mask_path_dir, split_path):
    split_names = ["train", "val", "test"]
    image_paths, label_paths, mask_paths, filenames = read_split_files(img_path_dir, label_path_dir, mask_path_dir, split_path, split_names)

    datasets = {}
    for split in split_names:
        datasets[split] = create_dataset(image_paths[split], label_paths[split], mask_paths[split], filenames[split])

    dataset = DatasetDict(datasets)

    return dataset

def craft_cv_datasetdict(img_path_dir, label_path_dir, mask_path_dir, split_path):
    split_names = ["train", "val"]
    image_paths, label_paths, mask_paths, filenames = read_split_files(img_path_dir, label_path_dir, mask_path_dir, split_path, split_names)

    # combine train and val for cross-validation
    cv_image_paths = image_paths["train"] + image_paths["val"]
    cv_label_paths = label_paths["train"] + label_paths["val"]
    cv_mask_paths = mask_paths["train"] + mask_paths["val"]
    cv_filenames = filenames["train"] + filenames["val"]

    cv_train_dataset = create_dataset(cv_image_paths, cv_label_paths, cv_mask_paths, cv_filenames)

    return DatasetDict({"cv_train": cv_train_dataset, "val": None, "test": None})

def craft_labelled_dataset(image_dir, label_dir, mask_dir):
    image_paths, label_paths, mask_paths, filenames = [], [], [], []
    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)
    masks = os.listdir(mask_dir)
    length = len(images)
    assert length == len(labels) == len(masks), "Number of files in directories do not match"

    for i in range(length):
        filenames.append(images[i].split('.')[0])
        image_paths.append(os.path.join(image_dir, images[i]))
        label_paths.append(os.path.join(label_dir, labels[i]))
        mask_paths.append(os.path.join(mask_dir, masks[i]))

    return create_dataset(image_paths, label_paths, mask_paths, filenames)



class Dinov2forSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, 32, 32, config.num_labels)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        # use frozen features
        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        # get the patch embeddings - so we exclude the CLS token
        # print(f"last hidden layer: {outputs.last_hidden_state.shape}")
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        # print(f"patch embeddings: {patch_embeddings.shape}")

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode='bilinear', align_corners=False)

        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze().float())

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)
    