import sys
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.losses import iou_loss, modified_mean_squared_error
from src.data_loader import is_valid_bbox, create_bbox_dataset


class BBoxModel:
    def __init__(
        self,
        input_shape,
        max_boxes,
        normalize=True,
        augmentations=["none", "horizontal_flip", "vertical_flip", "rotate"],
        model_fn=None,
        model_filepath=None,
    ):
        self.input_shape = input_shape
        self.max_boxes = max_boxes
        self.normalize = normalize
        self.augmentations = augmentations
        self.norm_mean = None
        self.norm_std = None
        self.unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

        if model_filepath:
            self.model = BBoxModel.load(model_filepath)
        else:
            self.model = model_fn(input_shape, max_boxes) if model_fn else self._build_model(input_shape, max_boxes)

    def _build_model(self, img_shape, max_boxes):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=img_shape),
            
            # Encoder: Convolutional layers
            tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
            tf.keras.layers.ELU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
            tf.keras.layers.ELU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(256, (3, 3), padding="same"),
            tf.keras.layers.ELU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Decoder: Convolution for bounding box regression
            tf.keras.layers.Conv2D(512, (3, 3), padding="same"),
            tf.keras.layers.ELU(),
            
            # Final convolutional layer for predicting bounding boxes
            tf.keras.layers.Conv2D(4 * max_boxes, (1, 1), padding="same"),
            tf.keras.layers.ELU(),
            
            # Global Average Pooling to reduce spatial dimensions
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Reshape to (batch_size, max_boxes, 4)
            tf.keras.layers.Reshape((max_boxes, 4))  # We want a fixed number of bounding boxes per image
        ])
        return model

    def compile(
        self,
        optimizer=Adam(learning_rate=0.0001),
        loss=iou_loss,
        metrics=["mae", "accuracy"],
    ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_normalization_constants(self, dataset):
        if self.norm_mean is not None and self.norm_std is not None:
            return tf.constant(self.norm_mean, dtype=tf.float32), tf.constant(self.norm_std, dtype=tf.float32)

        sum_pixels = tf.zeros((16,), dtype=tf.float32)
        sum_squares = tf.zeros((16,), dtype=tf.float32)
        num_pixels = tf.Variable(0, dtype=tf.int32)

        for image, _ in dataset:
            num_pixels.assign_add(tf.reduce_prod(tf.shape(image)[:-1]))  # Total pixels across batch
            sum_pixels += tf.reduce_sum(image, axis=[0, 1])  # Sum across height & width
            sum_squares += tf.reduce_sum(tf.square(image), axis=[0, 1])  # Sum of squares

        mean = sum_pixels / tf.cast(num_pixels, tf.float32)
        variance = (sum_squares / tf.cast(num_pixels, tf.float32)) - tf.square(mean)
        stddev = tf.sqrt(variance)

        self.norm_mean = tf.constant(mean).numpy()
        self.norm_std = tf.constant(stddev).numpy()

        print(f"Normalization constants: mean={self.norm_mean}, stddev={self.norm_std}")

        return mean, stddev

    def preprocess_dataset(self, dataset: tf.data.Dataset):
        mean, std = self.get_normalization_constants(dataset) if self.normalize else (0.0, 1.0)

        # resize images
        dataset = dataset.map(lambda img, lab: (tf.image.resize(img, self.input_shape[:-1]), lab))

        # normalize images
        if self.normalize:
            dataset = dataset.map(lambda img, lab: (BBoxModel.normalize_image(img, mean, std), lab))

        # augment images
        dataset = dataset.flat_map(lambda img, label: BBoxModel.augment_dataset(img, label, self.augmentations))
        dataset = dataset.map(lambda img, lab: (img, BBoxModel.normalize_bbox(lab, self.input_shape)))

        return dataset

    @staticmethod
    def normalize_image(image, mean, std):
        return (image - mean) / std
    
    @staticmethod
    def normalize_bbox(bbox, img_shape):
        height = tf.cast(img_shape[0], tf.float32)
        width = tf.cast(img_shape[1], tf.float32)

        # Normalize bounding box coordinates
        bbox = tf.stack([
            bbox[..., 0] / width,   # x-left
            bbox[..., 1] / width,   # x-right
            bbox[..., 2] / height,  # y-top
            bbox[..., 3] / height   # y-bottom
        ], axis=-1)

        return bbox

    @staticmethod
    def augment_image(image, bboxes, transformation):
        if transformation == "none":
            return image, bboxes

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

    @staticmethod
    def augment_dataset(
        image,
        bbox,
        augmentations=["none", "horizontal_flip", "vertical_flip", "rotate"],
    ):
        datasets = []
        for augmentation in augmentations:
            img, box = BBoxModel.augment_image(image, bbox, augmentation)
            datasets.append(tf.data.Dataset.from_tensors((img, box)))
        return tf.data.Dataset.from_tensor_slices(datasets).flat_map(lambda x: x)

    def train(self, train_dataset, epochs=10, batch_size=8):
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return self.model.fit(train_dataset, epochs=epochs)

    def evaluate(self, test_data):
        return self.model.evaluate(test_data)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, output_dir):
        self.model.save(f"{output_dir}/{self.unique_id}_bbox_model.h5")
        attrs_dict = {k: self.__dict__[k] for k in self.__dict__ if k != "model"}
        attrs_dict["norm_mean"] = self.norm_mean.tolist() if self.norm_mean is not None else None
        attrs_dict["norm_std"] = self.norm_std.tolist() if self.norm_std is not None else None

        with open(f"{output_dir}/{self.unique_id}_attrs.yaml", "w") as attrs_file:
            yaml.safe_dump(attrs_dict, attrs_file)

    @classmethod
    def load(cls, model_attrs_file: str):
        model_attrs = yaml.safe_load(open(model_attrs_file))
        model_file = model_attrs_file.replace("_attrs.yaml", "_bbox_model.h5")

        obj = cls(
            input_shape=model_attrs['input_shape'],
            max_boxes=model_attrs['max_boxes'],
            normalize=model_attrs['normalize'],
            augmentations=model_attrs['augmentations'],
        )

        obj.norm_mean = model_attrs.get("norm_mean", None)
        obj.norm_std = model_attrs.get("norm_std", None)

        obj.norm_mean = np.array(obj.norm_mean) if obj.norm_mean is not None else None
        obj.norm_std = np.array(obj.norm_std) if obj.norm_std is not None else None

        obj.model = load_model(model_file, custom_objects={"iou_loss": iou_loss, "modified_mean_squared_error": modified_mean_squared_error, 'ciou_loss': ciou_loss})
        return obj


if __name__ == "__main__":
    config_file = sys.argv[1]

    config_dict = yaml.safe_load(open(config_file))

    data_dir = config_dict.get("data_dir", "./data/raw_data/STARCOP_train_easy")
    max_boxes = config_dict.get("max_boxes", 10)
    image_shape = config_dict.get("image_shape", (512, 512))
    normalize = config_dict.get("normalize", False)
    augmentations = config_dict.get("augmentations", ["none", "horizontal_flip", "vertical_flip", "rotate"])

    exclude_dirs = config_dict.get("exclude_dirs", [])

    batch_size = config_dict.get("batch_size", 8)
    epochs = config_dict.get("epochs", 10)

    model = BBoxModel((*image_shape, 16), max_boxes, normalize=normalize, augmentations=augmentations)
    model.compile()
    print(model.model.summary())
    print("Model created successfully.")
    train_dataset = create_bbox_dataset(data_dir, max_boxes=max_boxes, exclude_dirs=exclude_dirs)
    train_dataset = model.preprocess_dataset(train_dataset)
    model.train(train_dataset, epochs=epochs, batch_size=batch_size)

    model.save('./logs')
