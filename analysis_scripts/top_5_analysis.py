
import os
import pandas as pd

datasets = ['raw', 'morph', 'otsu']
architectures = ['unet_brain', 'unet_smp', 'pspnet', 'deeplabv3plus', 'deeplabv3', 'dinov2']
# create a list of test files


# Create a dictionary for top 5 samples of each model with each dataset (NO ONES WHERE SIC = 0)
# Find the best performing model for each sample, and record

for dataset in datasets:
    for architecture in architectures:
        print(f"Dataset: {dataset}, Architecture: {architecture}")
        csv_file = os.path.abspath(os.path.join(f"./test_data_output/{architecture}/{dataset}", "evaluation_scores.csv"))
        df = pd.read_csv(csv_file)
        df['Sample'] = df['Sample'].astype(str)
        df = df[df['SIC Label'] != 0]
        df = df.sort_values(by='F1 Score', ascending=False)
        df = df.head(5)
        print(df)
        print()

data_list = []
for model in architectures:
    for dataset in datasets:




import pandas as pd
import numpy as np

# Sample data generation
num_images = 300
image_ids = [f'Img{i}' for i in range(1, num_images + 1)]
models = [f'Model_{chr(65 + i)}' for i in range(6)]  # Model_A to Model_F
datasets = [f'Dataset_{i}' for i in range(1, 4)]     # Dataset_1 to Dataset_3

# Create a list for each column
data_list = []
for model in models:
    for dataset in datasets:
        for image_id in image_ids:
            data_list.append({
                'Image_ID': image_id,
                'Model': model,
                'Dataset': dataset,
                'IoU': np.random.rand(),
                'Dice': np.random.rand(),
                'F1_Score': np.random.rand()
            })

# Create DataFrame
df = pd.DataFrame(data_list)

# Set MultiIndex
df.set_index(['Model', 'Dataset', 'Image_ID'], inplace=True)

# Reset index for grouping
df_reset = df.reset_index()

# Group by 'Model' and 'Dataset'
grouped = df_reset.groupby(['Model', 'Dataset'])

# Define the function to get top 5 images
def get_top_n_images(group, metric='IoU', n=5):
    return group.nlargest(n, metric)

# Apply the function to get top 5 images for each group
top5_images = grouped.apply(get_top_n_images, metric='IoU', n=5).reset_index(drop=True)

# Set MultiIndex back if needed
top5_images.set_index(['Model', 'Dataset', 'Image_ID'], inplace=True)

# Display top 5 images for a specific model/dataset
model = 'Model_C'
dataset = 'Dataset_2'
print(f"Top 5 images for {model} on {dataset}:")
print(top5_images.loc[(model, dataset)])
