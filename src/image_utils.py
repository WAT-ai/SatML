import os
import numpy as np
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt
from ipywidgets import interact
import math
from keras_cv import losses

def read_tiff_from_file(file_path: str | os.PathLike) -> np.ndarray:
    """
    reads a tiff file to a numpy array
    Assumes file exists
    
    Args:
        file_path (str | os.PathLike): 

    Returns:
        np.ndarray: numpy array containing file contents. Assumes BGR format
    """
    pass # TODO: Finish this function


def plot_tiff_images(dir: str | os.PathLike) -> None:
    """
    Display all .tif images in a directory using a slider

    Args:
        dir (str | os.PathLike)

    Returns:
        None
    """

    dir_path = f"data/raw_data/STARCOP_train_easy/{dir}" # NOTE: Modularize to work with other datasets as needed
    files_to_exclude = ['label_rgba.tif', 'labelbinary.tif', 'mag1c.tif', 'weight_mag1c.tif']

    if os.path.isdir(dir_path): # Ensure the directory exists
        total_files = 0
        images = []
        for file in os.listdir(dir_path):
            if file in files_to_exclude: # Skip files to exclude
                continue

            total_files += 1

            file_path = os.path.join(dir_path, file)
            try:
                img = Image.open(file_path)
                images.append((file, img)) # Store both filename and img object
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        print(f"Extracted {len(images)}/{total_files} images from {dir} directory.")

        def show_image(idx):
            plt.figure(figsize=(5, 5))
            plt.imshow(images[idx][1], cmap='gray')
            plt.title(f"Image {idx + 1} of {len(images)}: {images[idx][0]}")
            plt.axis('off')
            plt.show()

        # Create a slider for displaying images
        interact(show_image, idx=(0, len(images) - 1))

    else:
        print("Unable to access the provided directory.")


"""
Calculates the varonRatio between two bands S and B. S is the signal band, B 
is the background band which is a band with the same dimensions as S, but has
values from a wavelength without methane absorption, both bands store methane 
absorption levels in a 2D array. The goal is to be able to compare these two bands 
by returning the mean and std deviation.
Requires: B and S are the same dimensions, B != 0
"""
def varonRatio(S, B, c):

    ratio = (c * S - B)/B 

    mean = np.mean(ratio)
    std_deviation = np.std(ratio)

    return mean, std_deviation

"""
Generates a bounding box around each blob present within a binary label mask.
The label mask may contain 0-n blobs, where n is the number of blobs present in the mask.

Args:
    label_mask (np.ndarray): 2D numpy array of 0s and 1s

Returns:
    np.ndarray: numpy array of bounding boxes, where each bounding box is represented 
    as a tuple of (x-left, x-right, y-top, y-bottom).
"""

def binary_bbox(label_mask):
    assert label_mask.ndim == 2, "labelMask must be a 2D numpy array"
    
    labelled_arr = measure.label(label_image=label_mask, connectivity=2)
    bboxes = []
    
    for region in measure.regionprops(labelled_arr):
        min_row, min_col, max_row, max_col = region.bbox
        bboxes.append((min_col, (max_col - 1), min_row, (max_row - 1)))
    
    return np.array(bboxes)


    
def compare_bbox(true_bbox: tuple|list, pred_bbox: tuple|list, metric: str = "iou") -> float:
    """ Wrapper function for iou_metrics function, verifying bounding boxes and metric.

    IoU is the basic metric used to find amount of overlap between bounding boxes. 
    
    GIoU improves IoU by considering distance between boxes when they don't overlap. 
        
    CIoU is DIoU except adding the ability to consider the aspect ratio of the bounding boxes.
    
    Args:
        true_bbox:
            2d-list|2d-np nparray for the true bounding box, 
            with format [[t_left, t_top, t_right, t_bot], [...], ...]
        pred_bbox: 
            2d-list|2d-np nparray for the predicted bounding box. 
            with format [[p_left, p_top, p_right, p_bot], [...], ...]
        metrics: one of ["iou", "giou", "ciou"]:
        
    Returns:
        float: The similarity loss between true and pred bounding box. 
            iou_loss will return a value between 0 and 1. 
            other losses will return a value between 0 and 2.
            The lower the better.
       
    """
    # check for correct type
    if metric not in ["iou", "giou", "ciou"]:
        raise ValueError(
        'Unknown type {}, not iou/diou'.format(metric))
    
    # Convert inputs to NumPy arrays if they are not already
    true_bbox = np.array(true_bbox) if not isinstance(true_bbox, np.ndarray) else true_bbox
    pred_bbox = np.array(pred_bbox) if not isinstance(pred_bbox, np.ndarray) else pred_bbox
    
    # Ensure inputs are 2D arrays
    if true_bbox.ndim != 2 or pred_bbox.ndim != 2:
        raise ValueError("Bounding boxes must be 2D arrays.")
    
    # Ensure each bounding box has four coordinates (x_min, y_min, x_max, y_max)
    if true_bbox.shape[1] != 4 or pred_bbox.shape[1] != 4:
        raise ValueError("Each bounding box must have exactly four coordinates.")
    
    # Validate each bounding box
    for bbox in true_bbox:
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(f"Invalid bounding box: {bbox} in true_bbox")
    for bbox in pred_bbox:
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(f"Invalid bounding box: {bbox} in pred_bbox")
    
    # Ensure bounding boxes contain valid numeric types (int or float)
    if not (np.issubdtype(true_bbox.dtype, np.integer) or np.issubdtype(true_bbox.dtype, np.floating)):
        raise TypeError("true_bbox must contain int or float values.")
    if not (np.issubdtype(pred_bbox.dtype, np.integer) or np.issubdtype(pred_bbox.dtype, np.floating)):
        raise TypeError("pred_bbox must contain int or float values.")
    
    # get the loss function values
    iou_value = None
    if metric == 'iou':
        loss = losses.IoULoss("XYXY", "linear")
        iou_value = loss(true_bbox, pred_bbox).numpy()
    if metric == 'giou': 
        loss = losses.GIoULoss("XYXY")
        iou_value = loss(true_bbox, pred_bbox).numpy()
    if metric == 'ciou':
        loss = losses.CIoULoss("XYXY")
        iou_value = loss(true_bbox, pred_bbox).numpy()
        
    return iou_value

if __name__ == '__main__':
    res = compare_bbox([[1, 1, 3, 4]], [[900, 400, 1000, 800]], "giou")
    print(res)