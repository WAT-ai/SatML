import os
import cv2
import numpy as np


def read_tiff_from_file(file_path: str | os.PathLike) -> np.ndarray:
    """reads a tiff file to a numpy array
    Assumes file exists

    Args:
        file_path (str | os.PathLike): 

    Returns:
        np.ndarray: numpy array containing file contents. Assumes BGR format
    """
    return cv2.imread(file_path)
