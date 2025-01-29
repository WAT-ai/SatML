import argparse
from models.UNetModule import UNet
from src.data_loader import create_dataset

def train_segmentation(train_dataset, test_dataset, input_channels, num_classes, epochs, output_path):
    """Script to train U-Net segmentation model given numpy arrays containing train and test images and labels.

    Images are in n, h, w, c format (float32).
    Labels are in h, w format (uint8).
    """

    # Train model
    unet = UNet(input_channels, num_classes)
    unet.train(train_dataset, test_dataset, epochs=epochs)
    unet.model.save_weights(f'{output_path}/checkpoint.weights.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', type=str, default='./data/raw_data/STARCOP_train_easy')
    parser.add_argument('-output_path', '--output_path', type=str, default='./logs')
    parser.add_argument('-input_channels', '--input_channels', type=int, default=16)
    parser.add_argument('-num_classes', '--num_classes', type=int, default=2)
    parser.add_argument('-epochs', '--epochs', type=int, default=20)
    args = parser.parse_args()

    # Load the complete dataset
    dataset = create_dataset(args.data_path)
    
    # Determine the total number of samples in the dataset
    total_samples = sum(1 for _ in dataset)
    
    # Calculate split sizes (Using default 80/20 split for testing)
    train_size = int(total_samples * 0.8)
    test_size = total_samples - train_size

    # Split the dataset into training and testing subsets
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)  

    # Train the model
    train_segmentation(train_dataset, test_dataset, args.input_channels, args.num_classes, args.epochs, args.output_path)