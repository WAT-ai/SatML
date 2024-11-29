import os
import argparse
import numpy as np
import tensorflow as tf
from UNetModule import UNet

def train_segmentation(train_x, train_y, test_x, test_y, input_channels, num_classes, epochs, output_path):
    """Script to train U-Net segmentation model given numpy arrays containing train and test images and labels.

    Images are in n, h, w, c format (float32).
    Labels are in h, w format (uint8).
    """

    # Correct shape of label arrays (n, h, w) -> (n, h, w, 1)
    train_y_chan = np.expand_dims(train_y, axis=3)
    test_y_chan = np.expand_dims(test_y, axis=3)

    # Create tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices({"image": train_x, "segmentation_mask": train_y_chan})
    test_dataset = tf.data.Dataset.from_tensor_slices({"image": test_x, "segmentation_mask": test_y_chan})

    # Train model
    unet = UNet(input_channels, num_classes)
    unet.train(train_dataset, test_dataset, epochs=epochs)
    unet.model.save_weights(f'{output_path}/checkpoint.weights.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', type=str, default='/home/sarahmakki12/SatML/data/np_data')
    parser.add_argument('-output_path', '--output_path', type=str, default='/home/sarahmakki12/SatML/logs')
    parser.add_argument('-input_channels', '--input_channels', type=int, default=16)
    parser.add_argument('-num_classes', '--num_classes', type=int, default=2)
    parser.add_argument('-epochs', '--epochs', type=int, default=20)
    args = parser.parse_args()

    # Load datasets
    train_x = np.load(f'{args.data_path}/train_x.npy')
    train_y = np.load(f'{args.data_path}/train_y.npy')
    test_x = np.load(f'{args.data_path}/test_x.npy')
    test_y = np.load(f'{args.data_path}/test_y.npy')

    # Train model
    train_segmentation(train_x, train_y, test_x, test_y, args.input_channels, args.num_classes, args.epochs, args.output_path)