import tensorflow as tf

from processor.BaseProcessor import BaseProcessor
from src.image_utils import is_valid_bbox

class BoundingBoxProcessor(BaseProcessor):

    def __init__(self, config, input_shape, normalize=True, augmentations=None):
        super().__init__(config)
        self.input_shape = input_shape
        self.normalize = normalize
        self.augmentations = augmentations if augmentations else []
        self.norm_mean = None
        self.norm_std = None

    def get_normalization_constants(self, dataset):
        if not self.normalize:
            return (0.0, 1.0)
        
        if self.norm_mean is not None and self.norm_std is not None:
            return tf.constant(self.norm_mean, dtype=tf.float32), tf.constant(self.norm_std, dtype=tf.float32)

        sum_pixels = tf.zeros((16,), dtype=tf.float32)
        sum_squares = tf.zeros((16,), dtype=tf.float32)
        num_pixels = tf.Variable(0, dtype=tf.int32)

        for image_batch, _ in dataset:
            num_pixels.assign_add(tf.reduce_prod(tf.shape(image_batch)[:-1]))  # Total pixels across batch
            sum_pixels += tf.reduce_sum(image_batch, axis=[0, 1, 2])  # Sum across height & width
            sum_squares += tf.reduce_sum(tf.square(image_batch), axis=[0, 1, 2])  # Sum of squares

        self.norm_mean = sum_pixels / tf.cast(num_pixels, tf.float32)
        variance = (sum_squares / tf.cast(num_pixels, tf.float32)) - tf.square(self.norm_mean)
        self.norm_std = tf.sqrt(variance)

        print(f"Normalization constants: mean={self.norm_mean}, stddev={self.norm_std}")
    
    def resize(self, dataset):
        return dataset.map(lambda img, lab: (tf.image.resize(img, self.input_shape[:-1]), lab))

    def normalize_dataset(self, dataset):
        if self.normalize:
            return dataset.map(lambda img, lab: ((img - self.norm_mean) / self.norm_std, lab))

    def augment_dataset(self, dataset):
        return dataset.flat_map(lambda img, bbox: tf.data.Dataset.from_tensor_slices(
            [self.augment_image(img, bbox, augmentation) for augmentation in self.augmentations]
        ))

    def augment_image(self, image, bboxes, transformation):
        augmented_bboxes = []
        valid_mask = tf.cast(tf.map_fn(is_valid_bbox, bboxes, dtype=tf.bool), tf.bool)
        valid_mask = tf.expand_dims(valid_mask, axis=-1)
        valid_mask = tf.broadcast_to(valid_mask, tf.shape(bboxes))
        image_shape = tf.cast(tf.shape(image), tf.float32)

        if transformation == "horizontal_flip":
            image = tf.image.flip_left_right(image)
            augmented_bboxes = tf.where(
                valid_mask,
                tf.stack(
                    [
                        image_shape[1] - bboxes[:, 1] - 1,
                        image_shape[1] - bboxes[:, 0] - 1,
                        bboxes[:, 2],
                        bboxes[:, 3],
                    ],
                    axis=1,
                ),
                tf.fill(tf.shape(bboxes), -1.0),
            )
        elif transformation == "vertical_flip":
            image = tf.image.flip_up_down(image)
            augmented_bboxes = tf.where(
                valid_mask,
                tf.stack(
                    [
                        bboxes[:, 0],
                        bboxes[:, 1],
                        image_shape[0] - bboxes[:, 3] - 1,
                        image_shape[0] - bboxes[:, 2] - 1,
                    ],
                    axis=1,
                ),
                tf.fill(tf.shape(bboxes), -1.0),
            )
        elif transformation == "rotate":
            image = tf.image.rot90(image)
            augmented_bboxes = tf.where(
                valid_mask,
                tf.stack(
                    [
                        bboxes[:, 2],
                        bboxes[:, 3],
                        image_shape[1] - bboxes[:, 1] - 1,
                        image_shape[1] - bboxes[:, 0] - 1,
                    ],
                    axis=1,
                ),
                tf.fill(tf.shape(bboxes), -1.0),
            )
        return image, augmented_bboxes

    def normalize_bbox(self, dataset):
        height = tf.cast(self.input_shape[0], tf.float32)
        width = tf.cast(self.input_shape[1], tf.float32)

        # Normalize bounding box coordinates
        return dataset.map(lambda img, bbox: (img, tf.stack([
            bbox[..., 0] / width,   # x-left
            bbox[..., 1] / width,   # x-right
            bbox[..., 2] / height,  # y-top
            bbox[..., 3] / height   # y-bottom
        ], axis=-1)))
