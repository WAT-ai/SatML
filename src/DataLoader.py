import tensorflow as tf
from enum import Enum
from src.image_utils import data_generator, bbox_data_generator


class DatasetType(Enum):
    SEGMENTATION = "segmentation"
    BOUNDING_BOX = "bounding_box"


class DataLoader:
    def __init__(self, data_dir, dataset_type: DatasetType, batch_size=32, max_boxes=10, exclude_dirs=[]):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.max_boxes = max_boxes
        self.exclude_dirs = exclude_dirs
        self.dataset = self._create_dataset()

    def _create_dataset(self):
        """
        Creates a dataset based on the dataset type.
        """
        if self.dataset_type == DatasetType.SEGMENTATION:
            return self._create_segmentation_dataset()
        elif self.dataset_type == DatasetType.BOUNDING_BOX:
            return self._create_bbox_dataset()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _create_segmentation_dataset(self):
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
            tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32)    # Labels
        )

        dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(self.data_dir),
            output_signature=output_sig
        )

        # Transform dataset to dictionary format
        dataset = dataset.map(
            lambda img, lbl: {"image": img, "segmentation_mask": lbl},
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return self._apply_transforms(dataset)

    def _create_bbox_dataset(self):
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
            lambda: bbox_data_generator(self.data_dir, self.max_boxes, self.exclude_dirs),
            output_signature=output_sig
        )

        return self._apply_transforms(dataset)

    def _apply_transforms(self, dataset):
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_dataset(self):
        return self.dataset
