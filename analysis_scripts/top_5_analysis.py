
import os
import pandas as pd
import matplotlib.pyplot as plt

datasets = ['raw', 'morph', 'otsu']
architectures = ['unet', 'deeplabv3plus', 'segformer']

# Create a dictionary for top 5 samples of each model with each dataset (NO ONES WHERE SIC = 0)
# Find the best performing model for each sample, and record

# Create a list for each column
data_list = []
for model in architectures:
    for dataset in datasets:

        csv_file = os.path.abspath(os.path.join(f"./test_data_output/{model}/{dataset}", "evaluation_scores.csv"))
        df = pd.read_csv(csv_file)
        df['Sample'] = df['Sample'].astype(str)
        df = df[df['SIC Label'] != 0]
        image_ids = df['Sample'].unique()
        df.set_index('Sample', inplace=True)

        for image_id in image_ids:
            data_list.append({
                'Image_ID': image_id,
                'Model': model,
                'Dataset': dataset,
                'BCE Loss': df.loc[image_id]['BCE Loss'],
                'IoU': df.loc[image_id]['IOU'],
                'Dice': df.loc[image_id]['Dice Coefficient'],
                'F1_Score': df.loc[image_id]['F1 Score'],
                'Pixel Accuracy': df.loc[image_id]['Pixel Accuracy'],
                'Foreground Accuracy': df.loc[image_id]['Foreground Accuracy'],
                'True Positive': df.loc[image_id]['Number True Positive'],
                'False Positive': df.loc[image_id]['Number False Positive'],
                'Sea Ice Concentration': df.loc[image_id]['SIC Label'],
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

def get_bottom_n_images(group, metric='IoU', n=5):
    return group.nsmallest(n, metric)

# Apply the function to get top 5 images for each group
top5_images_iou = grouped.apply(get_top_n_images, metric='IoU', n=5).reset_index(drop=True)
bottom5_images_iou = grouped.apply(get_bottom_n_images, metric='IoU', n=5).reset_index(drop=True)

# Set MultiIndex back if needed
top5_images_iou.set_index(['Model', 'Dataset'], inplace=True)
bottom5_images_iou.set_index(['Model', 'Dataset'], inplace=True)


def create_best_worst_dict(df_top, df_bottom):
    best_worst_dict = {}
    for model in architectures:
        for dataset in datasets:
            best_worst_dict[(model, dataset)] = {
                'best': df_top.loc[(model, dataset)]['Image_ID'].unique(),
                'worst': df_bottom.loc[(model, dataset)]['Image_ID'].unique()
            }
    return best_worst_dict

best_worst_dict = create_best_worst_dict(top5_images_iou, bottom5_images_iou)


def view_top_n_images(model, dataset, best_worst_dict):

    top_images = best_worst_dict[(model, dataset)]['best']
    worst_images = best_worst_dict[(model, dataset)]['worst']

    img_dir = f"/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/{dataset}"

    fig, axs = plt.subplots(5, 1, figsize=(5, 10))

    for i, image_id in enumerate(top_images):
        img = plt.imread(os.path.join(img_dir, f"{image_id}.png"))
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f"Top {i+1} Image: {image_id}")

    plt.show()

    fig, axs = plt.subplots(5, 1, figsize=(10, 5))

    for i, image_id in enumerate(worst_images):
        img = plt.imread(os.path.join(img_dir, f"{image_id}.png"))
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f"Worst {i+1} Image: {image_id}")

    plt.show()

# view_top_n_images('segformer', 'morph', best_worst_dict)

# Find intersections of best and worst images
