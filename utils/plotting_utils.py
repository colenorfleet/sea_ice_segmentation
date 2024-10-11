
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2





def save_segmentation_image(image, target, prediction, filename, img_output_path):

    image = image[0]
    target = target.squeeze().detach().cpu().numpy()
    prediction = prediction.squeeze().detach().cpu().numpy()

    pixel_classification = plot_pixel_classification(prediction, target)

    target_img = (target * 255).astype(np.uint8)
    prediction_img = (prediction * 255).astype(np.uint8)

    # save prediction image
    prediction_path = os.path.join(img_output_path, 'prediction')
    comparison_path = os.path.join(img_output_path, 'comparison')
    os.makedirs(prediction_path, exist_ok=True)
    os.makedirs(comparison_path, exist_ok=True)
    cv2.imwrite(os.path.join(prediction_path, filename[0] + ".png"), prediction_img)

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

    plt.savefig(os.path.join(comparison_path, filename[0] + ".png"))
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


def create_mask(image, topo, type):

    if type == 'raw':
        raw_topo_mask = np.where(topo > 0, 1, 0).astype('uint8')
        return raw_topo_mask

    elif type == 'morph':
        raw_topo_mask = np.where(topo > 0, 1, 0).astype('uint8')
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph_binary_ice_mask = cv2.morphologyEx(raw_topo_mask.astype('uint8'), cv2.MORPH_CLOSE, close_kernel, iterations=1)
        return morph_binary_ice_mask

    elif type == 'otsu':
        raw_topo_mask = np.where(topo > 0, 1, 0).astype('uint8')
        thresholded_gray_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        binary_otsu_mask = np.where(thresholded_gray_image > 0, 1, 0)
        binary_ice_mask = np.where((raw_topo_mask + binary_otsu_mask) > 1, 1, 0)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        otsu_binary_ice_mask = cv2.morphologyEx(binary_ice_mask.astype('uint8'), cv2.MORPH_CLOSE, close_kernel, iterations=1)
        return otsu_binary_ice_mask

    else:
        raise ValueError("Invalid mask type. Must be 'raw', 'morph', or 'otsu'.")