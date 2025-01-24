import os
import tensorflow as tf
from src.image_utils import data_generator

def create_dataset(dir: str | os.PathLike) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset with images and labels grouped in dictionary format as given:
        - {"image": image_data, "segmentation_mask": label_data}
        - "image": (512, 512, 16, 1) in float32.
        - "segmentation_mask": (512, 512) in uint8.

    Args:
        dir (str | os.PathLike): Path to the directory containing the data.

    Returns:
        tf.data.Dataset: A TensorFlow dataset.
    """
    output_sig = (
        tf.TensorSpec(shape=(512, 512, 16, 1), dtype=tf.float32),  # Images
        tf.TensorSpec(shape=(512, 512), dtype=tf.uint8)           # Labels
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(dir),
        output_signature=output_sig
    )

    # Transform the dataset to key-val format: {"image": image, "segmentation_mask": label}
    dataset = dataset.map(
        lambda img, lbl: {"image": img, "segmentation_mask": lbl},
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


if __name__ == "__main__":
    # Test the create_dataset function
    train_data_path = './data/raw_data/STARCOP_train_easy'
    dataset = create_dataset(train_data_path)

    # Fetch a few samples from the dataset
    for i, data_point in enumerate(dataset.take(3)):  # Verify first 3 samples
        print(f"Sample {i + 1}:")
        print("Keys:", data_point.keys())
        print("Image shape:", data_point["image"].shape)
        print("Segmentation mask shape:", data_point["segmentation_mask"].shape)
        print()
