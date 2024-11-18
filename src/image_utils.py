import os
import numpy as np
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt
from ipywidgets import interact
from typing import List, Tuple, Generator
import math

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


def load_image_set(dir: str | os.PathLike, file_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Extract all .tif images and their labels data from a given directory

    Args:
        dir (str | os.PathLike): 
        file_names: list[str]: A list of image file names (frequencies) to extract

    Raises:
        FileNotFoundError: raised if directory cannot be found

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            - A list of the images stored as numpy float32 arrays
            - A list of label data stored as numpy float32 arrays
    '''
    img_labels = ['labelbinary.tif']  # Image label files to extract

    if os.path.isdir(dir):
        images, labels = [], []
        for file in os.listdir(dir):
            if file in file_names or file in img_labels:
                file_path = os.path.join(dir, file)
                try:
                    with Image.open(file_path) as img:
                        data = np.array(img, dtype=np.float32) # Store img/labels as float32 type array
                        if file in file_names:
                            images.append(data)
                        else:
                            labels.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        return np.array(images), np.array(labels)
    else:
        raise FileNotFoundError(f"Unable to find the {dir} directory.")
    

def data_generator(dir: str | os.PathLike) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    file_names = [
        "TOA_AVIRIS_460nm.tif",
        "TOA_AVIRIS_550nm.tif",
        "TOA_AVIRIS_640nm.tif",
        "TOA_AVIRIS_2004nm.tif",
        "TOA_AVIRIS_2109nm.tif",
        "TOA_AVIRIS_2310nm.tif",
        "TOA_AVIRIS_2350nm.tif",
        "TOA_AVIRIS_2360nm.tif",
        "TOA_WV3_SWIR1.tif",
        "TOA_WV3_SWIR2.tif",
        "TOA_WV3_SWIR3.tif",
        "TOA_WV3_SWIR4.tif",
        "TOA_WV3_SWIR5.tif",
        "TOA_WV3_SWIR6.tif",
        "TOA_WV3_SWIR7.tif",
        "TOA_WV3_SWIR8.tif"
    ]
    if os.path.isdir(dir):
        for entry in os.listdir(dir):
            sub_dir = os.path.join(dir, entry)
            if os.path.isdir(sub_dir):
                images, labels = load_image_set(sub_dir, file_names)
                yield images, labels
    else:
        raise FileNotFoundError(f"Unable to find the {dir} directory.")



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

def iou_metrics(true_bbox: tuple|list , pred_bbox: tuple|list , metric: str = "iou") -> float:
    """ Computing the specified IoU metric between two bounding boxes 
    
    Args:
        true_bbox:
            tuple|list for the true bounding box, 
            with format (t_left, t_right, t_top, t_bot) 
        pred_bbox: 
            tuple|list for the predicted bounding box. 
            with format ( p_left, p_right, p_top, p_bot)
        metrics: one of ["iou", "giou", "diou", "ciou"]:
        
    Returns:
        float: The similarity between true and pred bounding box. 
            iou will return a value between 0 and 1. 
            others will return a value between -1 and 1
    """
    
    #unpack values
    t_left, t_right, t_top, t_bot = true_bbox
    p_left, p_right, p_top, p_bot = pred_bbox
    
    # Calculate intersection area
    inter_x_l = max(t_left , p_left)
    inter_x_r = min(t_right , p_right)
    inter_y_t = max(t_top , p_top)
    inter_y_b = min(t_bot , p_bot)
    interArea = max(0, inter_x_r - inter_x_l) * max(0, inter_y_b - inter_y_t)
    
    # Calculate area of both boxes
    true_area = max(0, t_right - t_left) * max(0, t_bot - t_top)
    pred_area = max(0, p_right - p_left) * max(0, p_bot - p_top)
    union_area =  (true_area + pred_area - interArea)
    
    # Calculate iou 
    iou = interArea / union_area if union_area != 0 else 0  
    
    if metric == 'iou':
        return iou
    
    # Find bounds of minimum sized box to enclose both area
    enclose_x_l = min(t_left , p_left)
    enclose_x_r = max(t_right , p_right)
    enclose_y_b = max(t_bot , p_bot)
    enclose_y_t = min(t_top, p_top)
   
    # calculate giou. 
    if metric == 'giou': 
        encloseArea = max(0, enclose_x_r - enclose_x_l) * max(0, enclose_y_b - enclose_y_t )
        giou = iou - ( (encloseArea - union_area) / encloseArea) if encloseArea != 0 else iou
        return giou
    
    # Calculate the euclidean distnace between each box's centers, and also the diagonal of enclosed box
    pred_center = np.array([(p_top + p_bot) / 2 , (p_left + p_right) / 2])
    true_center =  np.array([(t_top + t_bot) / 2 , (t_left + t_right) / 2])
    euclidean_dist = np.linalg.norm(true_center - pred_center)
    enclose_diag =  np.linalg.norm([enclose_y_b - enclose_y_t , enclose_x_r - enclose_x_l  ])
    
    # calculate diou
    diou = iou - ((euclidean_dist ** 2 ) / (enclose_diag ** 2)) if enclose_diag != 0 else iou
    if metric == 'diou':
        return diou
    
    # calculate ciou
    if metric == 'ciou':
        # get width and heights of bounding boxes and calculate ratios
        pred_width = p_right - p_left
        pred_height = p_bot - p_top
        true_width = t_right - t_left
        true_height = t_bot - t_top
        pred_ratio = pred_width / pred_height if pred_height != 0 else 0
        true_ratio = true_width / true_height if true_height != 0 else 0
        
        # Calculate v and alpha values for aspect ratio
        arctan = math.atan(pred_ratio) - math.atan(true_ratio)
        v = 4 * ((arctan / math.pi)**2)
        alpha = v / ((1 - iou) + v) if ((1 - iou) + v) != 0 else 0
        
        return diou - alpha * v
    
def compare_bbox(true_bbox: tuple|list, pred_bbox: tuple|list, metric: str = "iou") -> float:
    """ Wrapper function for iou_metrics function,verifying bounding boxes and metric.

    IoU is the basic metric used to find amount of overlap between bounding boxes. 
    
    GIoU improves IoU by considering distance between boxes when they don't overlap. 
    
    DIoU considers vertical and horizontal orientation and often converges faster than the pior two. 
    
    CIoU is DIoU except adding the ability to consider the aspect ratio of the bounding boxes.
    
    Args:
        true_bbox:
            tuple|list for the true bounding box, 
            with format (t_left, t_right, t_top, t_bot) 
        pred_bbox: 
            tuple|list for the predicted bounding box. 
            with format ( p_left, p_right, p_top, p_bot)
        metrics: one of ["iou", "giou", "diou", "ciou"]:
        
    Returns:
        float: The similarity between true and pred bounding box. 
            iou will return a value between 0 and 1. 
            others will return a value between -1 and 1.
            The higher the better.
       
    """
    # check for correct type
    if metric not in ["iou", "giou", "diou", "ciou"]:
        raise ValueError(
        'Unknown type {}, not iou/diou'.format(metric))
    
    # make sure boxes are of length
    assert len(true_bbox) == 4
    assert len(pred_bbox) == 4
    
    # unpack values
    t_left, t_right, t_top, t_bot = true_bbox
    p_left, p_right, p_top, p_bot = pred_bbox
    
    # Check valid values of bounding box
    if t_left >= t_right or t_top >= t_bot:
        raise ValueError('Invalid bounding box: true_bbox')
    if p_left >= p_right or p_top >= p_bot:
        raise ValueError('Invalid bounding box: pred_bbox')
    
    # Check for valid input format
    if not all(isinstance(coord, (int, float)) for coord in true_bbox + pred_bbox):
        raise ValueError("All coordinates in true_bbox and pred_bbox must be either int or float.")
    
    iou_value = iou_metrics(true_bbox, pred_bbox, metric)
    return iou_value

