import tensorflow as tf

from src.data_loader.BaseDataLoader import BaseDataLoader
from src.image_utils import bbox_data_generator

class BoundingBoxDataLoader(BaseDataLoader):
    def __init__(self, dataset_dir, batch_size=32, max_boxes=10, exclude_dirs=[]):
        super().__init__(dataset_dir, batch_size, exclude_dirs)
        self.max_boxes = max_boxes 

    def create_dataset(self):
        """
        Creates a TensorFlow dataset with images and their bounding box labels

        Returns:
            tf.data.Dataset: Dataset with images and their bounding box labels
            - Images: (512, 512, 16)
            - Labels: (max_boxes, 4)
        """
        output_sig = (
            tf.TensorSpec(shape=(512, 512, 16), dtype=tf.float32),  # Images
            tf.TensorSpec(shape=(self.max_boxes, 4), dtype=tf.float32)  # Bounding boxes
        )

        dataset = tf.data.Dataset.from_generator(
            lambda: bbox_data_generator(self.dataset_dir, self.max_boxes, self.exclude_dirs),
            output_signature=output_sig
        )

        self.dataset = dataset