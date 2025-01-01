import tensorflow as tf
from src.image_utils import data_generator

def create_dataset() -> tf.data.Dataset: # TODO: Modify to accept other base data dirs
    """
    Creates a TensorFlow dataset with images and their bounding box labels:
        - Images: (512, 512, 16)
        - Labels: (max_boxes, 4)
    Each image channel corressponds to a specific hyperspectral frequency image.
    """
    output_sig = (
        tf.TensorSpec(shape=(512, 512, 16), dtype=tf.float32),  # Images
        tf.TensorSpec(shape=(10, 4), dtype=tf.float32)   # bounding box labels
    )

    dir = './data/raw_data/STARCOP_train_easy' # Use train easy dataset by default

    return tf.data.Dataset.from_generator(
        lambda: data_generator(dir),
        output_signature=output_sig
    )


if __name__ == "__main__":
    dataset = create_dataset()

    for images, labels in dataset.take(1):  # Take one batch of data
        print(f'Images shape: {images.shape}')
        print(f'Labels shape: {labels.shape}')
