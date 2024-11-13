import os
import argparse
import numpy as np
import tensorflow as tf
from UNetModule import UNet

def train_segmentation(train_x, train_y, test_x, test_y, output_channels, epochs):
    """Script to train U-Net segmentation model given numpy arrays containing train and test images and labels.

    Images are in n, h, w, c format (float32).
    Labels are in h, w format (uint8).
    """
    # Normalize images
    x_max = max(np.max(train_x), np.max(test_x))
    x_min = min(np.min(train_x), np.min(test_x))

    train_x_norm = (train_x - x_min) / (x_max - x_min)
    test_x_norm = (test_x - x_min) / (x_max - x_min)

    # Correct shape of label arrays (n, h, w) -> (n, h, w, 1)
    train_y_chan = np.expand_dims(train_y, axis=3)
    test_y_chan = np.expand_dims(test_y, axis=3)

    # Create tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices({"image": train_x_norm, "segmentation_mask": train_y_chan})
    test_dataset = tf.data.Dataset.from_tensor_slices({"image": test_x_norm, "segmentation_mask": test_y_chan})

    # Train model
    unet = UNet(output_channels)
    unet.train(train_dataset, test_dataset, epochs=epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', type=str, default='/SatML/data/np_data')
    parser.add_argument('-output_channels', '--output_channels', type=int, default=16)
    parser.add_argument('-epochs', '--epochs', type=int, default=20)
    args = parser.parse_args()

    # Load datasets
    train_x = np.load(f'{args.data_path}/train_x.npy')
    train_y = np.load(f'{args.data_path}/train_y.npy')
    test_x = np.load(f'{args.data_path}/test_x.npy')
    test_y = np.load(f'{args.data_path}/test_y.npy')

    # Train model
    train_segmentation(train_x, train_y, test_x, test_y, args.output_channels, args.epochs)