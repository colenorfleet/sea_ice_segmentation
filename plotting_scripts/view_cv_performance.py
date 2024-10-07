
import os 
import sys
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

input_dict = {'loss': ['Avg BCE Train Loss', 'Avg BCE Val Loss'],
              'iou': ['Avg Train IOU', 'Avg Val IOU'],
                'f1': ['Avg Train F1', 'Avg Val F1'],
}

# Folder Structure
# cv_output
    # model_name
        # dataset_name
            # best_ice_seg_model_fold_{0-4}.pth
            # training_logs_fold_{0-4}.csv
