import tensorflow as tf
from image_utils import data_generator

def create_dataset() -> tf.data.Dataset: # TODO: Modify to accept other base data dirs
    output_sig = (
        tf.TensorSpec(shape=(16, 512, 512), dtype=tf.float32),  # Images
        tf.TensorSpec(shape=(1, 512, 512), dtype=tf.float32)  # Labels
    )

    dir = './data/raw_data/STARCOP_train_easy' # Use train easy dataset by default

    return tf.data.Dataset.from_generator(
        lambda: data_generator(dir),
        output_signature=output_sig
    )


dataset = create_dataset()

for images, labels in dataset.take(1):  # Take one batch of data
    print(f'Images shape: {images.shape}')
    print(f'Labels shape: {labels.shape}')
