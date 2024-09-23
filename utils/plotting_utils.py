
import matplotlib.pyplot as plt
import numpy as np
import os





def save_segmentation_image(image, target, prediction, filename, img_output_path):

    image = image[0]
    target = target.squeeze().detach().cpu().numpy()
    prediction = prediction.squeeze().detach().cpu().numpy()

    pixel_classification = plot_pixel_classification(prediction, target)

    target_img = (target * 255).astype(np.uint8)
    prediction_img = (prediction * 255).astype(np.uint8)

    # 

    fig = plt.figure(figsize=(30,15), layout='tight')

    ax0 = fig.add_subplot(141)
    ax0.imshow(image)
    ax0.set_title('Image')
    ax0.axis('off')

    ax1 = fig.add_subplot(142)
    ax1.imshow(target_img, cmap='gray')
    ax1.set_title('Target')
    ax1.axis('off')

    ax2 = fig.add_subplot(143)
    ax2.imshow(prediction_img, cmap='gray')
    ax2.set_title('Prediction')
    ax2.axis('off')

    ax3 = fig.add_subplot(144)
    ax3.imshow(pixel_classification)
    ax3.set_title('Pixel Classification')
    ax3.axis('off')

    plt.savefig(os.path.join(img_output_path, filename[0] + ".png"))
    plt.close(fig)


    return



def plot_pixel_classification(pred_mask, true_mask):

    """
    Plot the classification of each pixel into TP, FP, TN, FN using different colors.
    
    TP: Green
    FP: Red
    TN: Blue
    FN: Yellow
    """

    assert type(pred_mask) == np.ndarray, "pred_mask must be a numpy array"
    assert type(true_mask) == np.ndarray, "true_mask must be a numpy array"

    # create a blank canvas
    height, width = pred_mask.shape
    result = np.zeros((height, width, 3), dtype=np.uint8)

    # define colors
    colors = {
        "TP": [0, 255, 0], # green
        "FP": [255, 0, 0], # red
        "TN": [0, 0, 255], # blue
        "FN": [255, 255, 0], # yellow
    }

    # classify each pixel
    for i in range(height):
        for j in range(width):
            if pred_mask[i, j] == 1 and true_mask[i, j] == 1:
                result[i, j] = colors["TP"]
            elif pred_mask[i, j] == 1 and true_mask[i, j] == 0:
                result[i, j] = colors["FP"]
            elif pred_mask[i, j] == 0 and true_mask[i, j] == 0:
                result[i, j] = colors["TN"]
            elif pred_mask[i, j] == 0 and true_mask[i, j] == 1:
                result[i, j] = colors["FN"]


    return result