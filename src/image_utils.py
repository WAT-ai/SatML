import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional

def read_tiff_from_file(file_path: str | os.PathLike) -> np.ndarray:
    """
    reads a tiff file to a numpy array
    Assumes file exists
    
    Args:
        file_path (str | os.PathLike): 

    Returns:
        np.ndarray: numpy array containing file contents. Assumes BGR format
    """
    return cv2.imread(file_path)
    


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

def varon_iteration(dir_path: str, output_file: str, c_threshold: float, num_images: int, num_folders: Optional[int]=None, pixels: Optional[int]= 255):
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
    
    image_file_names = [
        "TOA_AVIRIS_460nm.tif", 
        "TOA_AVIRIS_550nm.tif", 
        "TOA_AVIRIS_640nm.tif",
        "TOA_AVIRIS_2004nm.tif",
        "TOA_AVIRIS_2004nm.tif",
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
    
    if num_folders is not None:
        all_folders = all_folders[:num_folders]

    for image_folder in all_folders:
        folder_path = os.path.join(dir_path, image_folder)
        
        if not os.path.isdir(folder_path):
            continue

        hyperspectral_images = []
        image_prime_sums = []
        for i in range(num_images):
            img_path = os.path.join(folder_path, image_file_names[i])
            
            with Image.open(img_path) as img_data:
                img_array = np.array(img_data)
                img_subset = img_array[:pixels, :pixels]
                hyperspectral_images.append(img_subset)
                image_prime_sums.append(np.sum(remove_outliers_with_zscore(img_subset, c_threshold)))
            

        
        current_matrix = np.zeros((num_images, num_images)) 

        for k in range(num_images):
            S = hyperspectral_images[k]
            S_prime_sum = image_prime_sums[k]
            for j in range(k, num_images): 
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
    
    
  
