import tensorflow as tf
from image_utils import data_generator

def create_dataset() -> tf.data.Dataset: # TODO: Modify to accept other base data dirs
    output_sig = (
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # Image
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # Label 1
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32) # Label 2
    )

    dir = './data/raw_data/STARCOP_train_easy' # Use train easy dataset by default

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(dir),
        output_signature=output_sig
    )

    return dataset

dataset = create_dataset()