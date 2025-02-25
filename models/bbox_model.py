import sys
import yaml
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.losses import iou_loss, modified_mean_squared_error, ciou_loss
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
        blank_percentage=0.1
    ):
        self.input_shape = input_shape
        self.max_boxes = max_boxes
        self.normalize = normalize
        self.augmentations = augmentations
        self.blank_percentage = blank_percentage
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

            # Encoder
            tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('swish'),  # Using Swish activation
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),  # Slightly higher dropout as depth increases

            tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            # Bottleneck
            tf.keras.layers.Flatten(),

            # Dense Layers for Bounding Box Regression
            tf.keras.layers.Dense(128, activation="gelu"),  # Using GELU activation
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),  # High dropout to regularize fully connected layers

            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(32, activation="gelu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Bounding Box Output
            tf.keras.layers.Dense(4 * max_boxes, activation="sigmoid"),
            tf.keras.layers.Reshape((max_boxes, 4))
        ])
        return model

    def compile(
        self,
        optimizer=Adam(learning_rate=0.001),
        loss=ciou_loss,
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
        def is_valid_sample(image, bboxes):
            """Returns True if the image has at least one valid bounding box."""
            valid_mask = tf.map_fn(is_valid_bbox, bboxes, dtype=tf.bool)
            return tf.reduce_any(valid_mask)

        def is_invalid_sample(image, bboxes):
            """Returns True if the image has only invalid bounding boxes."""
            return tf.logical_not(is_valid_sample(image, bboxes))

        mean, std = self.get_normalization_constants(dataset) if self.normalize else (0.0, 1.0)

        dataset = dataset.map(lambda img, lab: (img, BBoxModel.normalize_bbox(lab, img.shape)))

        # resize images
        dataset = dataset.map(lambda img, lab: (tf.image.resize(img, self.input_shape[:-1]), lab))

        dataset = dataset.flat_map(lambda img, label: BBoxModel.augment_dataset(img, label, self.augmentations))

        # normalize images
        if self.normalize:
            dataset = dataset.map(lambda img, lab: (BBoxModel.normalize_image(img, mean, std), lab))

        # filter invalid samples
        # Split datasets
        valid_ds = dataset.filter(is_valid_sample)
        if self.blank_percentage > 0:
            invalid_ds = dataset.filter(is_invalid_sample)

            # Balance datasets with controlled ratio
            dataset = tf.data.Dataset.sample_from_datasets(
                [valid_ds, invalid_ds], weights=[1 - self.blank_percentage, self.blank_percentage], seed=42, stop_on_empty_dataset=True
            )
        else:
            dataset = valid_ds

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
            tf.round(bbox[..., 0] / width * 100) / 100,   # x-left
            tf.round(bbox[..., 1] / width * 100) / 100,   # x-right
            tf.round(bbox[..., 2] / height * 100) / 100,  # y-top
            tf.round(bbox[..., 3] / height * 100) / 100   # y-bottom
        ], axis=-1)

        return bbox

    @staticmethod
    def augment_image(image, bboxes, transformation, max_translate=0.3, crop_fraction=0.6):
        augmented_bboxes = []
        valid_mask = tf.cast(tf.map_fn(is_valid_bbox, bboxes, dtype=tf.bool), tf.bool)
        has_valid_boxes = tf.reduce_any(valid_mask)

        if transformation == "none":
            return (image, bboxes) if has_valid_boxes else (image, tf.zeros_like(bboxes))


        if not has_valid_boxes:
            return image, tf.zeros_like(bboxes)


        valid_mask = tf.expand_dims(valid_mask, axis=-1)
        valid_mask = tf.broadcast_to(valid_mask, tf.shape(bboxes))

        if transformation == "horizontal_flip":
            image = tf.image.flip_left_right(image)
            augmented_bboxes = tf.where(
                valid_mask,
                tf.stack(
                    [
                        1 - bboxes[:, 1],
                        1 - bboxes[:, 0],
                        bboxes[:, 2],
                        bboxes[:, 3],
                    ],
                    axis=1,
                ),
                tf.fill(tf.shape(bboxes), 0.0),
            )
        elif transformation == "vertical_flip":
            image = tf.image.flip_up_down(image)
            augmented_bboxes = tf.where(
                valid_mask,
                tf.stack(
                    [
                        bboxes[:, 0],
                        bboxes[:, 1],
                        1 - bboxes[:, 3],
                        1 - bboxes[:, 2],
                    ],
                    axis=1,
                ),
                tf.fill(tf.shape(bboxes), 0.0),
            )
        elif transformation == "rotate":
            image = tf.image.rot90(image)
            augmented_bboxes = tf.where(
                valid_mask,
                tf.stack(
                    [
                        1 - bboxes[:, 3],
                        1 - bboxes[:, 2],
                        bboxes[:, 0],
                        bboxes[:, 1],
                    ],
                    axis=1,
                ),
                tf.fill(tf.shape(bboxes), 0.0),
            )
        elif transformation == "translate":
            # Generate random translation offsets within the max_translate range
            dx = random.uniform(-max_translate, max_translate)
            dy = random.uniform(-max_translate, max_translate)

           # Convert to pixel shifts
            shift_x = int(dx * tf.cast(tf.shape(image)[1], tf.float32))
            shift_y = int(dy * tf.cast(tf.shape(image)[0], tf.float32))

            # Pad and crop the image to simulate translation
            paddings = [[tf.maximum(shift_y, 0), tf.maximum(-shift_y, 0)], [tf.maximum(shift_x, 0), tf.maximum(-shift_x, 0)], [0, 0]]
            image = tf.pad(image, paddings, mode='CONSTANT', constant_values=0)
            image = tf.image.crop_to_bounding_box(
                image,
                offset_height=tf.maximum(-shift_y, 0),
                offset_width=tf.maximum(-shift_x, 0),
                target_height=tf.shape(image)[0] - tf.abs(shift_y),
                target_width=tf.shape(image)[1] - tf.abs(shift_x)
            )

            # Adjust bounding boxes
            augmented_bboxes = tf.where(
                valid_mask,
                tf.stack(
                    [
                        tf.clip_by_value(bboxes[:, 0] + tf.cast(dx, tf.float32), 0.0, tf.cast(1.0, tf.float32)),
                        tf.clip_by_value(bboxes[:, 1] + tf.cast(dx, tf.float32), 0.0, tf.cast(1.0, tf.float32)),
                        tf.clip_by_value(bboxes[:, 2] + tf.cast(dy, tf.float32), 0.0, tf.cast(1.0, tf.float32)),
                        tf.clip_by_value(bboxes[:, 3] + tf.cast(dy, tf.float32), 0.0, tf.cast(1.0, tf.float32)),
                    ],
                    axis=1,
                ),
                tf.fill(tf.shape(bboxes), 0.0),
            )
        elif transformation == "crop":
            img_h, img_w, _ = tf.unstack(tf.shape(image))

            # Determine crop size
            crop_h = tf.cast(tf.round(crop_fraction * tf.cast(img_h, tf.float32)), tf.int32)
            crop_w = tf.cast(tf.round(crop_fraction * tf.cast(img_w, tf.float32)), tf.int32)

            # Randomly select top-left corner for cropping
            max_y_offset = img_h - crop_h
            max_x_offset = img_w - crop_w
            offset_y = tf.random.uniform([], 0, max_y_offset, dtype=tf.int32, seed=42)
            offset_x = tf.random.uniform([], 0, max_x_offset, dtype=tf.int32, seed=42)

            # Crop image
            image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, crop_h, crop_w)
            # Resize back to original size
            image = tf.image.resize(image, [img_h, img_w])

            # Adjust bounding boxes
            augmented_bboxes = tf.stack([
                (bboxes[:, 0] - tf.cast(offset_x, tf.float32) / tf.cast(img_w, tf.float32)) / crop_fraction,
                (bboxes[:, 1] - tf.cast(offset_x, tf.float32) / tf.cast(img_w, tf.float32)) / crop_fraction,
                (bboxes[:, 2] - tf.cast(offset_y, tf.float32) / tf.cast(img_h, tf.float32)) / crop_fraction,
                (bboxes[:, 3] - tf.cast(offset_y, tf.float32) / tf.cast(img_h, tf.float32)) / crop_fraction,
            ], axis=1)

            # Ensure bounding boxes are within valid range
            augmented_bboxes = tf.clip_by_value(augmented_bboxes, 0.0, 1.0)

        return image, augmented_bboxes

    @staticmethod
    def filter_invalid_sample(image, bboxes, threshold=0.5):
        """
        Returns True if the sample contains enough valid bounding boxes.

        Args:
            image (tf.Tensor): The image tensor.
            bboxes (tf.Tensor): The bounding boxes tensor.
            threshold (float): Minimum proportion of valid bounding boxes.

        Returns:
            tf.bool: Whether the sample should be kept.
        """
        valid_mask = tf.map_fn(is_valid_bbox, bboxes, dtype=tf.bool)
        valid_count = tf.reduce_sum(tf.cast(valid_mask, tf.float32))
        total_count = tf.cast(tf.shape(bboxes)[0], tf.float32)

        return valid_count / total_count >= threshold

    @staticmethod
    def augment_dataset(
        image,
        bbox,
        augmentations=["none", "horizontal_flip", "vertical_flip", "rotate", "translate"],
    ):
        datasets = []
        for augmentation in augmentations:
            img, box = BBoxModel.augment_image(image, bbox, augmentation)
            datasets.append(tf.data.Dataset.from_tensors((img, box)))
        return tf.data.Dataset.from_tensor_slices(datasets).flat_map(lambda x: x)

    def train(self, dataset, epochs=10, batch_size=8):
        # Splitting the dataset for training and testing.
        def is_test(x, _):
            return x % 4 == 0

        def is_train(x, y):
            return not is_test(x, y)

        def recover(x, y):
            return y

        # Split the dataset for training.
        test_dataset = dataset.enumerate().filter(is_test).map(recover)

        # Split the dataset for testing/validation.
        train_dataset = dataset.enumerate().filter(is_train).map(recover)
        lr_patience = 20
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=lr_patience, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=lr_patience * 2.5, restore_best_weights=False),
        ]

        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
        self.model.fit(train_dataset, epochs=epochs, callbacks=callbacks, validation_data=test_dataset)

    def evaluate(self, test_data):
        return self.model.evaluate(test_data)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, output_dir):
        self.model.save(f"{output_dir}/{self.unique_id}_bbox_model.keras")
        attrs_dict = {k: self.__dict__[k] for k in self.__dict__ if k != "model"}
        attrs_dict["norm_mean"] = self.norm_mean.tolist() if self.norm_mean is not None else None
        attrs_dict["norm_std"] = self.norm_std.tolist() if self.norm_std is not None else None

        with open(f"{output_dir}/{self.unique_id}_attrs.yaml", "w") as attrs_file:
            yaml.safe_dump(attrs_dict, attrs_file)

        print(f"Model and attributes saved to {output_dir}: {self.unique_id}")

    @classmethod
    def load(cls, model_attrs_file: str):
        model_attrs = yaml.safe_load(open(model_attrs_file))
        model_file = model_attrs_file.replace("_attrs.yaml", "_bbox_model.keras")

        obj = cls(
            input_shape=model_attrs['input_shape'],
            max_boxes=model_attrs['max_boxes'],
            normalize=model_attrs['normalize'],
            augmentations=model_attrs['augmentations'],
            blank_percentage=model_attrs['blank_percentage']
        )

        obj.norm_mean = model_attrs.get("norm_mean", None)
        obj.norm_std = model_attrs.get("norm_std", None)

        obj.norm_mean = np.array(obj.norm_mean) if obj.norm_mean is not None else None
        obj.norm_std = np.array(obj.norm_std) if obj.norm_std is not None else None

        obj.model = load_model(
            model_file,
            custom_objects={
                "iou_loss": iou_loss,
                "modified_mean_squared_error": modified_mean_squared_error,
                'ciou_loss': ciou_loss
            },
            safe_mode=False,    # This has to be false to allow loading the lambda layer
        )
        return obj


if __name__ == "__main__":
    config_file = sys.argv[1]

    config_dict = yaml.safe_load(open(config_file))

    data_dir = config_dict.get("data_dir", "./data/raw_data/STARCOP_train_easy")
    max_boxes = config_dict.get("max_boxes", 10)
    image_shape = config_dict.get("image_shape", (512, 512))
    normalize = config_dict.get("normalize", False)
    augmentations = config_dict.get("augmentations", ["none", "horizontal_flip", "vertical_flip", "rotate"])
    blank_percentage = config_dict.get("blank_percentage", 0.1)

    exclude_dirs = config_dict.get("exclude_dirs", [])
    force_square = config_dict.get("force_square", False)

    batch_size = config_dict.get("batch_size", 8)
    epochs = config_dict.get("epochs", 10)

    model = BBoxModel((*image_shape, 16), max_boxes, normalize=normalize, augmentations=augmentations, blank_percentage=blank_percentage)
    model.compile()
    print(model.model.summary())
    print("Model created successfully.")
    train_dataset = create_bbox_dataset(data_dir, max_boxes=max_boxes, exclude_dirs=exclude_dirs, force_square=force_square)
    train_dataset = model.preprocess_dataset(train_dataset)
    model.train(train_dataset, epochs=epochs, batch_size=batch_size)

    model.save('./logs')
