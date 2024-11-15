import os
import cv2
import numpy as np
from PIL import Image

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

    z_scores = (flat_data - mean) / std_dev
    return flat_data[np.abs(z_scores) < threshold]

"""consumes a path to a directory (easy training), and name of the output file. For each
folder of images, it computes the varon ratio between each image creating
a c x c matrix where c is the number of hyperspectral images. Returns total mean and
standard deviation"""
def varon_iteration(dir_path, output_file, c_threshold, num_folders):
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

    num_images = 3

    all_folders = os.listdir(dir_path)
    
    if num_folders is not None:
        all_folders = all_folders[:num_folders]

    for image_folder in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, image_folder)
        
        if not os.path.isdir(folder_path):
            continue

        hyperspectral_images = []
        for i in range(num_images):
            img_path = os.path.join(folder_path, image_file_names[i])
            
            with Image.open(img_path) as img_data:
                img_array = np.array(img_data)
                hyperspectral_images.append(img_array) 

        
        current_matrix = np.zeros((num_images, num_images)) 

        for k in range(num_images):
            for j in range(k, num_images): 
                # I didn't do anything with the fact that S should be signal band and B should be background band
                S = hyperspectral_images[k]
                B = hyperspectral_images[j]

                # calculate c: note! S/B_prime are 1D arrays
                S_prime = np.array(remove_outliers_with_zscore(S, c_threshold))
                B_prime = np.array(remove_outliers_with_zscore(B, c_threshold))
                c = S_prime.sum()/B_prime.sum()

                mean, _ = varonRatio(S, B, c)
                current_matrix[k, j] = mean
                current_matrix[j, k] = mean

        final_matrix.append(current_matrix)

    np.save(output_file, final_matrix)
    overall_mean = np.mean(final_matrix)
    overall_stddev = np.std(final_matrix)

    return overall_mean, overall_stddev
    
    
  
