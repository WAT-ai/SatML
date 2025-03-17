import tensorflow as tf

from src.data_loader.base_data_loader import BaseDataLoader
from src.image_utils import data_generator

class SegmentationDataLoader(BaseDataLoader):
    def __init__(self, dataset_dir, batch_size=32, exclude_dirs=[]):
        super().__init__(dataset_dir, batch_size, exclude_dirs)

    def create_dataset(self):
        """
        Creates a TensorFlow dataset with images and labels grouped in dictionary format as given:
            - {"image": image_data, "segmentation_mask": label_data}
            - "image": (512, 512, 16) in float32.
            - "segmentation_mask": (512, 512, 1) in float32.

        Args:
            dir (str | os.PathLike): Path to the directory containing the data.

        Returns:
            tf.data.Dataset: A TensorFlow dataset.
        """
        output_sig = (
            tf.TensorSpec(shape=(512, 512, 16), dtype=tf.float32),  # Images
            tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32)    # Segmentation Masks
        )

        dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(self.dataset_dir),
            output_signature=output_sig
        )

        # Transform dataset to dictionary format
        dataset = dataset.map(
            lambda img, lbl: {"image": img, "segmentation_mask": lbl},
            num_parallel_calls=tf.data.AUTOTUNE
        )

        self.dataset = dataset
