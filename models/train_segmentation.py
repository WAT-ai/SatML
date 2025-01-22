import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from models.UNetModule import UNet
from models.cnn_model_factory import get_no_downsample_cnn_model
from src.image_utils import get_global_normalization_mean_std, resize_data_and_labels


# Generator function for training data
def train_generator(norm_mean, norm_std, train_x, train_y):
    def generator():
        for i in range(len(train_x)):
            image = train_x[i]
            label = train_y[i]

            yield (image - norm_mean) / norm_std, label

    return generator


# Generator function for testing data
def test_generator(norm_mean, norm_std, test_x, test_y):
    def generator():
        for i in range(len(test_x)):
            image = test_x[i]
            label = test_y[i]

            yield (image - norm_mean) / norm_std, label

    return generator


def train_unet(train_x, train_y, test_x, test_y, input_channels, num_classes, epochs, output_path):
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
    unet.model.summary()
    unet.train(train_dataset, test_dataset, epochs=epochs)
    unet.model.save_weights(f"{output_path}/{TIMESTAMP}_unet_checkpoint.weights.h5")


def train_no_downsample_cnn(
    train_x, train_y, test_x, test_y, input_channels, num_classes, epochs, output_path, resize_shape=(256, 256)
):
    """Script to train basic cnn segmentation model given numpy arrays containing train and test images and labels.

    Images are in n, h, w, c format (float32).
    Labels are in h, w format (uint8).
    """

    # limit train_x to where there are labels
    train_x = train_x[np.any(train_y, axis=(-1, -2))]
    train_y = train_y[np.any(train_y, axis=(-1, -2))]

    train_x, train_y = resize_data_and_labels(train_x, train_y[..., np.newaxis], resize_shape)
    test_x, test_y = resize_data_and_labels(test_x, test_y[..., np.newaxis], resize_shape)

    norm_mean, norm_std = get_global_normalization_mean_std(train_x)

    # Create tensorflow datasets
    train_dataset = tf.data.Dataset.from_generator(
        generator=train_generator(norm_mean[0], norm_std[0], train_x, train_y),
        output_signature=(
            tf.TensorSpec(shape=resize_shape + (16,), dtype=tf.float32),  # Images
            tf.TensorSpec(shape=resize_shape + (1,), dtype=tf.float32),  # Labels
        ),
    )

    test_dataset = tf.data.Dataset.from_generator(
        generator=test_generator(norm_mean[0], norm_std[0], test_x, test_y),
        output_signature=(
            tf.TensorSpec(shape=resize_shape + (16,), dtype=tf.float32),  # Images
            tf.TensorSpec(shape=resize_shape + (1,), dtype=tf.float32),  # Labels
        ),
    )

    batch_size = 32
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()

    # Train model
    data_shape = resize_shape + (input_channels,)
    model = get_no_downsample_cnn_model(data_shape, num_classes, 5.0)
    model.fit(train_dataset, validation_data=(test_dataset), epochs=epochs)
    model.save_weights(f"{output_path}/{TIMESTAMP}_nds_cnn_checkpoint.weights.h5")


if __name__ == "__main__":
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_type", "--model_type", type=str, default="nds_cnn")
    parser.add_argument("-data_path", "--data_path", type=str, default="/home/sarahmakki12/SatML/data/np_data")
    parser.add_argument("-output_path", "--output_path", type=str, default="/home/sarahmakki12/SatML/logs")
    parser.add_argument("-input_channels", "--input_channels", type=int, default=16)
    parser.add_argument("-num_classes", "--num_classes", type=int, default=2)
    parser.add_argument("-epochs", "--epochs", type=int, default=20)
    args = parser.parse_args()

    # Load datasets
    train_x = np.load(f"{args.data_path}/train_x.npy")
    train_y = np.load(f"{args.data_path}/train_y.npy")
    test_x = np.load(f"{args.data_path}/test_x.npy")
    test_y = np.load(f"{args.data_path}/test_y.npy")

    # Train model
    if args.model_type == "nds_cnn":
        train_no_downsample_cnn(
            train_x,
            train_y,
            test_x,
            test_y,
            args.input_channels,
            args.num_classes,
            args.epochs,
            args.output_path,
        )
    elif args.model_type == "unet":
        train_unet(
            train_x,
            train_y,
            test_x,
            test_y,
            args.input_channels,
            args.num_classes,
            args.epochs,
            args.output_path,
        )
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented.")
