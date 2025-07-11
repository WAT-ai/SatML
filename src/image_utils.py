import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from skimage import measure
from typing import Tuple, Generator, Optional, Union
from keras_cv import losses
from src.constants import IMAGE_FILE_NAMES


def read_tiff_from_file(file_path: str | os.PathLike) -> np.ndarray:
    """
    reads a tiff file to a numpy array
    Assumes file exists

    Args:
        file_path (str | os.PathLike):

    Returns:
        np.ndarray: numpy array containing file contents. Assumes BGR format
    """
    pass  # TODO: Finish this function


def plot_tiff_images(dir: str | os.PathLike) -> None:
    """
    Display all .tif images in a directory using a slider

    Args:
        dir (str | os.PathLike)

    Returns:
        None
    """

    dir_path = f"data/raw_data/STARCOP_train_easy/{dir}"  # NOTE: Modularize to work with other datasets as needed
    files_to_exclude = ["label_rgba.tif", "labelbinary.tif", "mag1c.tif", "weight_mag1c.tif"]

    if os.path.isdir(dir_path):  # Ensure the directory exists
        total_files = 0
        images = []
        for file in os.listdir(dir_path):
            if file in files_to_exclude:  # Skip files to exclude
                continue

            total_files += 1

            file_path = os.path.join(dir_path, file)
            try:
                img = Image.open(file_path)
                images.append((file, img))  # Store both filename and img object
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        print(f"Extracted {len(images)}/{total_files} images from {dir} directory.")

        def show_image(idx):
            plt.figure(figsize=(5, 5))
            plt.imshow(images[idx][1], cmap="gray")
            plt.title(f"Image {idx + 1} of {len(images)}: {images[idx][0]}")
            plt.axis("off")
            plt.show()

        # Create a slider for displaying images
        # interact(show_image, idx=(0, len(images) - 1))

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

    if std_dev != 0:
        z_scores = (flat_data - mean) / std_dev
    else:
        z_scores = 0

    return flat_data[np.abs(z_scores) < threshold]


def load_image_set(dir: str | os.PathLike, file_names: Tuple[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
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
    """
    img_labels = ["labelbinary.tif"]  # Image label files to extract

    if os.path.isdir(dir):
        images, labels = [], []
        for file in os.listdir(dir):
            if file in file_names or file in img_labels:
                file_path = os.path.join(dir, file)
                try:
                    with Image.open(file_path) as img:
                        data = np.array(img, dtype=np.float32)  # Store img/labels as float32 type array
                        if file in file_names:
                            images.append(data)
                        else:
                            labels.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        return np.stack(images, axis=-1), np.expand_dims(labels[0], axis=-1)  # Stack the image & labels array layerwise
    else:
        raise FileNotFoundError(f"Unable to find the {dir} directory.")


def is_valid_bbox(bbox: Union[np.ndarray, tf.Tensor]) -> bool:
    """Checks if a given bounding box is valid

    Args:
        bbox (np.ndarray): x-left, x-right, y-top, y-bottom coordinates
    """
    x_left, x_right, y_top, y_bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    if x_left >= x_right or y_top >= y_bottom:
        return False

    return True


def bbox_data_generator(
    dir: str | os.PathLike, max_boxes: int = 10, exclude_dirs: list = [], force_square: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Load images and generate bounding boxes for labels from subdirectories of a specified directory

    Args:
        dir (str | os.PathLike): Path to the base directory containing image subdirectories
        max_boxes (int): Maximum number of bounding boxes to generate per image
        exclude_dirs (list): List of subdirectories to exclude from processing
        force_square (bool): If True, adjust the bounding box to be square

    Yields:
        Generator[Tuple[np.ndarray, np.ndarray], None, None]:
            - A numpy array of images with shape (512, 512, 16)
            - A numpy array of bounding box labels with shape (max_boxes, 4)
            - Path to the sub_dir directory containing 16 images
    """

    if os.path.isdir(dir):
        entries = [sub_dir for sub_dir in os.listdir(dir) if sub_dir not in exclude_dirs]
        for entry in entries:
            sub_dir = os.path.join(dir, entry)
            if os.path.isdir(sub_dir):
                images, label_mask = load_image_set(sub_dir, IMAGE_FILE_NAMES)

                if label_mask.ndim > 2:
                    label_mask = np.squeeze(label_mask)

                # make surue its binary or integer type
                label_mask = label_mask.astype(int)

                # NOTE: get_single_bounding_box only generates one bounding box
                bboxes = [
                    bbox
                    for bbox in [get_single_bounding_box(label_mask, force_square=force_square)]
                    if is_valid_bbox(bbox)
                ]

                if len(bboxes) < max_boxes:
                    bboxes.extend([[0, 0, 0, 0]] * (max_boxes - len(bboxes)))

                yield images, np.array(bboxes, dtype=np.float32), sub_dir


def data_generator(dir: str | os.PathLike) -> Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
    """
    Load images, their labels, and the directory path from subdirectories of a specified directory.

    Args:
        dir (str | os.PathLike): Path to the base directory containing image subdirectories.

    Yields:
        Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
            - A numpy array of images with shape (512, 512, 16) as float32 format.
            - A numpy array of labels with shape (512, 512, 1) as float32 format.
            - Path to the sub_dir directory containing 16 images.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Notes:
        - The `file_names` list specifies the expected image files to be processed.
        - The `load_image_set` function is used to load and convert image and label data.
        - Each yielded tuple corresponds to a batch of images and labels from a subdirectory.
    """

    if os.path.isdir(dir):
        for entry in os.listdir(dir):
            sub_dir = os.path.join(dir, entry)
            if os.path.isdir(sub_dir):
                images, labels = load_image_set(sub_dir, IMAGE_FILE_NAMES)
                yield images, labels, sub_dir
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


def get_single_bounding_box(mask: np.ndarray, force_square: bool = True):
    """
    Generate a bounding box around a binary mask in a given 2D array.

    Parameters:
        mask (np.ndarray): A 2D numpy array representing the binary mask.
        force_square (bool): If True, adjust the bounding box to be square.

    Returns:
        np.array: An array of four integers (x_left, x_right, y_top, y_bottom).
    """
    # Validate input dimensions
    if mask.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    # Find non-zero elements in the mask
    non_zero_indices = np.argwhere(mask > 0)

    if non_zero_indices.size == 0:
        # No mask present
        return np.array([0, 0, 0, 0], dtype=np.float32)

    # Calculate the bounding box coordinates
    y_coords, x_coords = non_zero_indices[:, 0], non_zero_indices[:, 1]
    x_left = np.min(x_coords)
    x_right = np.max(x_coords)
    y_top = np.min(y_coords)
    y_bottom = np.max(y_coords)

    if force_square:
        width = x_right - x_left
        height = y_bottom - y_top
        side_length = max(width, height)

        # Center the square around the original bounding box
        x_center = (x_left + x_right) // 2
        y_center = (y_top + y_bottom) // 2

        half_side = side_length // 2
        x_left = max(0, x_center - half_side)
        x_right = x_left + side_length
        y_top = max(0, y_center - half_side)
        y_bottom = y_top + side_length

        # Adjust if the square goes beyond the mask dimensions
        if x_right >= mask.shape[1]:
            x_right = mask.shape[1] - 1
            x_left = x_right - side_length

        if y_bottom >= mask.shape[0]:
            y_bottom = mask.shape[0] - 1
            y_top = y_bottom - side_length

        # Ensure coordinates are within bounds
        x_left = max(0, x_left)
        y_top = max(0, y_top)
        x_right = min(mask.shape[1] - 1, x_right)
        y_bottom = min(mask.shape[0] - 1, y_bottom)

    return np.array([x_left, x_right, y_top, y_bottom], dtype=np.float32)


def compare_bbox(true_bbox: tuple | list, pred_bbox: tuple | list, metric: str = "iou") -> float:
    """Wrapper function for iou_metrics function, verifying bounding boxes and metric.

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
        raise ValueError("Unknown type {}, not iou/diou".format(metric))

    # # Convert inputs to NumPy arrays if they are not already
    # true_bbox = np.array(true_bbox) if not isinstance(true_bbox, np.ndarray) else true_bbox
    # pred_bbox = np.array(pred_bbox) if not isinstance(pred_bbox, np.ndarray) else pred_bbox

    # # Ensure inputs are 2D arrays
    # if true_bbox.ndim != 2 or pred_bbox.ndim != 2:
    #     raise ValueError("Bounding boxes must be 2D arrays.")

    # # Ensure each bounding box has four coordinates (x_min, y_min, x_max, y_max)
    # if true_bbox.shape[1] != 4 or pred_bbox.shape[1] != 4:
    #     raise ValueError("Each bounding box must have exactly four coordinates.")

    # # Validate each bounding box
    # for bbox in true_bbox:
    #     if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
    #         raise ValueError(f"Invalid bounding box: {bbox} in true_bbox")
    # for bbox in pred_bbox:
    #     if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
    #         raise ValueError(f"Invalid bounding box: {bbox} in pred_bbox")

    # # Ensure bounding boxes contain valid numeric types (int or float)
    # if not (np.issubdtype(true_bbox.dtype, np.integer) or np.issubdtype(true_bbox.dtype, np.floating)):
    #     raise TypeError("true_bbox must contain int or float values.")
    # if not (np.issubdtype(pred_bbox.dtype, np.integer) or np.issubdtype(pred_bbox.dtype, np.floating)):
    #     raise TypeError("pred_bbox must contain int or float values.")

    # get the loss function values
    iou_value = None
    if metric == "iou":
        loss = losses.IoULoss("XYXY", "linear")
        iou_value = loss(true_bbox, pred_bbox)
    if metric == "giou":
        loss = losses.GIoULoss("XYXY")
        iou_value = loss(true_bbox, pred_bbox)
    if metric == "ciou":
        loss = losses.CIoULoss("XYXY")
        iou_value = loss(true_bbox, pred_bbox)

    return iou_value


def varon_iteration(
    dir_path: str,
    c_threshold: float,
    num_bands: int,
    output_file: Optional[str] = None,
    images: Optional[list] = None,
    pixels: Optional[int] = 255,
) -> np.ndarray:
    """
    consumes a path to a directory (easy training), and name of the output file. For each
    folder of images, it computes the varon ratio between each image creating
    a c x c matrix where c is the number of hyperspectral images. Returns computed matrix


    Args:
        dir_path (str): path to directory with images
        c_threshold (float): z-threshold for outlier filtering
        num_bands (int): number of bands to process
        output_file (Optional[str], optional): name of the file to save the computed matrix. Computed matrix won't be saved if not specified. Defaults to None.
        images (Optional[list], optional): name of folders to process. Defaults to None.
        pixels (Optional[int], optional): number of pixels to process. Defaults to 255.
        save_path (Optional[str], optional): path to save the computed matrix. Matrix won't be saved if not specified. Defaults to None.
        If None, all folders will be processed

    Returns:
        np.ndarray: compute matrix containing varon ratios for all frequency channels
    """
    final_matrix = []

    folders_to_process = os.listdir(dir_path) if images is None else images

    for image_folder in folders_to_process:
        folder_path = os.path.join(dir_path, image_folder)

        if not os.path.isdir(folder_path):
            continue

        hyperspectral_images = []
        image_prime_sums = []
        for i in range(num_bands):
            img_path = os.path.join(folder_path, IMAGE_FILE_NAMES[i])

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
                if B_prime_sum != 0:
                    c = S_prime_sum / B_prime_sum
                else:
                    c = 1

                mean, _ = varonRatio(S, B, c)
                current_matrix[k, j] = mean
                current_matrix[j, k] = mean

        final_matrix.append(current_matrix)

    if output_file:
        np.save(output_file, final_matrix)

    return final_matrix


def get_global_normalization_mean_std(data):
    """Function to calculate the global mean and standard deviation of the data

    Args:
        data (np.ndarray): numpy array containing image data

    Returns:
        tuple: (np.ndarray, np.ndarray) containing the mean and standard deviation of the data
    """
    mean_global = np.mean(data, axis=(0, 1, 2), keepdims=True)
    std_global = np.std(data, axis=(0, 1, 2), keepdims=True)

    std_global[std_global == 0] = 1.0
    return mean_global, std_global


def resize_data_and_labels(x, y, reshape_size):
    """function to resize the data and labels to a specified size

    Args:
        x (np.ndarray or tf.tensor): image array
        y (np.ndarray or tf.tensor): label array
        reshape_size (tuple or list): shape to resize image and labels to

    Returns:
        tuple: (tf.tensor, tf.tensor): resized image and label arrays
    """
    x_resized = tf.image.resize(x, reshape_size)
    y_resized = tf.image.resize(y, reshape_size)

    return x_resized, y_resized
