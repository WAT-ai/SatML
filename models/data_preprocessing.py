# data loading and preprocessing
import numpy as np
from PIL import Image

def stack_images(image_paths):
    """
    Stacks multiple images to create a hyperspectral image
    Parameter:
        image_paths (list): List of file paths to the images
    Returns:
        np.ndarray: Stacked image data as a NumPy array with shape, (height, width, num_bands)
    """
    image_list = []

    for path in image_paths:
        # Open the image using PIL
        img = Image.open(path)
        # Convert the image to a NumPy array and add to the list
        image_list.append(np.array(img))

    # Stack images along a new axis to create a multi-band image
    stacked_image_data = np.stack(image_list, axis=-1)
    # stacked_image_data = np.expand_dims(stacked_image_data, axis=0)
    print("Stacked image shape:", stacked_image_data.shape)

    return stacked_image_data

def preprocess_images(image_data, target_size):
    """
    Resizes and normalizes the images.
    Paramaters:
        image_data (np.ndarray): The stacked image data.
        target_size (tuple): The desired size (height, width) for the images.
    Returns:
        np.ndarray: Preprocessed image data.
    """
    resized_images = []
    for i in range(image_data.shape[-1]):  
        img = Image.fromarray(image_data[:, :, i])
        img_resized = img.resize(target_size)
        resized_images.append(np.array(img_resized))

    resized_image_data = np.stack(resized_images, axis=-1)
    # normalized_image_data = resized_image_data / 255.0  

    return resized_image_data

def augment_images(image_data):
    """
    Augments the image data by flipping horizontally, vertically, and rotating the images 
    Parameter:
        image_data (np.ndarray): image data to augment
    Returns:
        np.ndarray: Augmented image data
    """
    augmented_images = [image_data] 

    flipped_horizontally = np.flip(image_data, axis=1)
    augmented_images.append(flipped_horizontally)

    flipped_vertically = np.flip(image_data, axis=0)
    augmented_images.append(flipped_vertically)

    # combine all augmented images into a single array
    augmented_data = np.concatenate(augmented_images, axis=0)

    return augmented_data