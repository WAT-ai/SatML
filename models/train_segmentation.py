import os
import argparse
import numpy as np
import tensorflow as tf
from UNetModule import UNet

def train_segmentation(output_channels, train_x, train_y, test_x, test_y):
    """Script to train U-Net segmentation model given numpy arrays containing train and test images and labels.

    Images are in n, h, w, c format (float32).
    Labels are in h, w format (uint8).
    """
    # Normalize images
    max = max(np.max(train_x), np.max(test_x))
    min = min(np.min(train_x), np.min(test_x))

    train_x_norm = (train_x - min) / (max - min)
    test_x_norm = (test_x - min) / (max - min)

    # Correct shape of label arrays
    train_y_chan = np.expand_dims(train_y, axis=3)
    test_y_chan = np.expand_dims(test_y, axis=3)

    # Create tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices({"image": train_x_norm, "segmentation_mask": train_y_chan})
    test_dataset = tf.data.Dataset.from_tensor_slices({"image": test_x_norm, "segmentation_mask": test_y_chan})

    # Train model
    unet = UNet(args.output_channels)
    unet.train(train_dataset, test_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', type=str, default='data/np_data')
    parser.add_argument('-output_channels', '--output_channels', type=int, default=16)
    args = parser.parse_args()

    # Load datasets
    train_x = np.load(f'{args.data_path}/train_x.npy')
    train_y = np.load(f'{args.data_path}/train_y.npy')
    test_x = np.load(f'{args.data_path}/test_x.npy')
    test_y = np.load(f'{args.data_path}/test_y.npy')

    # Train model
    train_segmentation(args.output_channels, train_x, train_y, test_x, test_y)