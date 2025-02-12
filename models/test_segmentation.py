import argparse
import json
import tensorflow as tf
from models.UNetModule import UNet
from src.data_loader import create_dataset

def evaluate_segmentation(model, test_dataset):
    """Evaluate the segmentation model on the test dataset.

    Args:
        model: Trained segmentation model.
        test_dataset: Dataset object for testing.
    
    Returns:
        metrics: A dictionary containing evaluation metrics and loss.
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    total_loss = 0
    total_samples = 0

    for images, labels in test_dataset:
        predictions = model(images, training=False)
        loss = loss_object(labels, predictions)
        total_loss += loss.numpy() * len(images)
        total_samples += len(images)
        metric_accuracy.update_state(labels, predictions)

    avg_loss = total_loss / total_samples
    accuracy = metric_accuracy.result().numpy()

    metrics = {
        "average_loss": avg_loss,
        "accuracy": accuracy,
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a segmentation model on a test dataset.")
    parser.add_argument("-model_type", "--model_type", type=str, default="unet", help="Type of the model (e.g., 'unet').")
    parser.add_argument("-model_path", "--model_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("-data_path", "--data_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("-output_path", "--output_path", type=str, default=None, help="Path to save evaluation results (optional).")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=16, help="Batch size for testing.")
    args = parser.parse_args()

    # Load the testing dataset
    test_dataset = create_dataset(args.data_path)
    test_dataset = test_dataset.batch(args.batch_size)

    # Load the model
    if args.model_type.lower() == "unet":
        # Assuming input_channels and num_classes can be inferred
        model = UNet(input_channels=16, num_classes=2)  # Modify these as per the specific model requirements
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.load_weights(args.model_path)

    # Evaluate the model
    metrics = evaluate_segmentation(model, test_dataset)

    # Print metrics to the console
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Optionally save metrics to a file
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Results saved to {args.output_path}")
