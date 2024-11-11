import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import interact
from typing import List, Tuple, Generator


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
                    data = np.array(Image.open(file_path), dtype=np.float32) # Store img/labels as float32 type array
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