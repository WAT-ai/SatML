import os
import numpy as np
from PIL import Image
from typing import Optional
from skimage import measure
import matplotlib.pyplot as plt
from ipywidgets import interact
from typing import List, Tuple, Generator
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

    ratio = np.where(B != 0, (c * S - B) / B, 0)
    mean = np.mean(ratio)
    std_deviation = np.std(ratio)

    return mean, std_deviation

def remove_outliers_with_zscore(data, threshold):
    flat_data = data.flatten()
    mean = np.mean(flat_data)
    std_dev = np.std(flat_data)

    if (std_dev != 0):
       z_scores = (flat_data - mean) / std_dev
    else:
       z_scores = 0

    return flat_data[np.abs(z_scores) < threshold]


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
            - Images stored as numpy float32 arrays with shape (512, 512, 16)
            - Labels data stored as numpy float32 arrays with shape (512, 512, 1)
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

        return np.stack(images, axis=-1), np.expand_dims(data, axis=-1) # Stack the image & labels array layerwise
    else:
        raise FileNotFoundError(f"Unable to find the {dir} directory.")
    

def data_generator(dir: str | os.PathLike) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Load images and their labels from subdirectories of a specified directory.

    Args:
        dir (str | os.PathLike): Path to the base directory containing image subdirectories.

    Yields:
        Generator[Tuple[np.ndarray, np.ndarray], None, None]: 
            - A numpy array of images with shape (512, 512, 16).
            - A numpy array of labels with shape (512, 512, 1).

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Notes:
        - The `file_names` list specifies the expected image files to be processed.
        - The `load_image_set` function is used to load and convert image and label data.
        - Each yielded tuple corresponds to a batch of images and labels from a subdirectory.
    """

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
        "TOA_WV3_SWIR8.tif"]

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

def createTestMatrix():

        data = np.array([
            [[0, -0.2104005415, -0.4138471552],
            [-0.2104005415, 0, -0.2586695576],
            [-0.4138471552, -0.2586695576, 0]],
            [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
            [[0, -0.2246186874, -0.3063050874],
            [-0.2246186874, 0, -0.1064981483],
            [-0.3063050874, -0.1064981483, 0]]
        ])

        np.save("tests/varon_correct.npy", data)

def varon_iteration(dir_path: str, output_file: str, c_threshold: float, num_bands: int, images: Optional[np.ndarray]=None, pixels: Optional[int]= 255):
    """
    consumes a path to a directory (easy training), and name of the output file. For each
    folder of images, it computes the varon ratio between each image creating
    a c x c matrix where c is the number of hyperspectral images. Returns computed matrix


    Args:
        dir_path (str): path to directory with images
        output_file (str): path to output file for computed matrix
        c_threshold (float): z-threshold for outlier filtering
        num_folders (Optional[int], optional): number of folders to process. Defaults to None.
        If None, all folders will be processed

    Returns:
        np.ndarray: compute matrix containing varon ratios for all frequency channels
    """
    final_matrix = []
    num_images = len(images)
    image_file_names = [
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
        "TOA_WV3_SWIR8.tif"]

    all_folders = os.listdir(dir_path)
    
    if num_images is not None:
        all_folders = all_folders[:num_images]

    for image_folder in all_folders:
        folder_path = os.path.join(dir_path, image_folder)
        
        if not os.path.isdir(folder_path):
            continue

        hyperspectral_images = []
        image_prime_sums = []
        for i in range(num_bands):
            img_path = os.path.join(folder_path, image_file_names[i])
            
            with Image.open(img_path) as img_data:
                img_array = np.array(img_data)
                img_subset = img_array[:pixels, :pixels]
                hyperspectral_images.append(img_subset)
                image_prime_sums.append(np.sum(remove_outliers_with_zscore(img_subset, c_threshold)))
            

        current_matrix = np.zeros((num_bands, num_bands)) 

        for k in range(num_bands):
            S = hyperspectral_images[k]
            S_prime_sum = image_prime_sums[k]
            for j in range(k, num_bands): 
                # I didn't do anything with the fact that S should be signal band and B should be background band
                B = hyperspectral_images[j]

                # calculate c: note! S/B_prime are 1D arrays
                B_prime_sum = image_prime_sums[j]
                if(B_prime_sum != 0):
                   c = S_prime_sum/B_prime_sum
                else:
                   c = 1

                mean, _ = varonRatio(S, B, c)
                current_matrix[k, j] = mean
                current_matrix[j, k] = mean

        final_matrix.append(current_matrix)

    np.save(output_file, final_matrix)

    return final_matrix
