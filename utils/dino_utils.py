
import torch
import torch.nn as nn
from torchvision import models


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=True)


    def forward(self, x):
        x = x.last_hidden_state[:, 1:, :]
        # print('seg head input size:', x.size())
        x = x.transpose(1, 2).reshape(x.size()[0], 768, 32, 32)
        # print('seg head reshaped size:', x.size())
        x = self.conv1(x)
        # print('seg head conv1 size:', x.size())
        x = self.relu(x)
        # print('seg head relu size:', x.size())
        x = self.conv2(x)
        # print('seg head conv2 size:', x.size())
        x = self.upsample(x) # upsample to the size of the input image
        # print('seg head upsample size:', x.size())
        return x
    

class DinoBinarySeg(nn.Module):
    def __init__(self, encoder, decoder):
        super(DinoBinarySeg, self).__init__()
        self.encoder = encoder
        self.decoder = decoder.decoder
        self.segmentation_head = decoder.segmentation_head
        self.channel_adjust = nn.Conv2d(768, 2048, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=True)


    def forward(self, x):
        # print('input size:', x.size())
        x = self.encoder(x)
        # print('encoder output size:', x.last_hidden_state.size()) 
        x = x.last_hidden_state[:, 1:, :]
        x = x.transpose(1, 2).reshape(x.size()[0], 768, 32, 32)
        # print('reshaped size:', x.size())

        # adjust channels 
        x = self.channel_adjust(x)
        x = self.decoder(x)
        x = self.segmentation_head(x)
        x = self.upsample(x)

        return x