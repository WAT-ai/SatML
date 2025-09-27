import sys
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.losses import iou_loss, modified_mean_squared_error, ciou_loss, yolo_dense_loss, yolo_focal_loss
from src.data_loader import has_valid_bbox, create_bbox_dataset
from src.constants import IMAGE_FILE_NAMES


class BBoxModel:
    def __init__(
        self,
        input_shape,
        max_boxes,
        normalize=True,
        augmentations=["none", "horizontal_flip", "vertical_flip", "rotate"],
        model_fn=None,
        model_filepath=None,
        blank_percentage=0.1,
        grid_size=16,
        boxes_per_grid=1,
    ):
        self.input_shape = input_shape
        self.max_boxes = max_boxes
        self.normalize = normalize
        self.augmentations = augmentations
        self.blank_percentage = blank_percentage
        self.norm_mean = None
        self.norm_std = None
        self.grid_size = grid_size
        self.boxes_per_grid = boxes_per_grid
        self.unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

        if model_filepath:
            self.model = BBoxModel.load(model_filepath)
        else:
            self.model = model_fn(input_shape, max_boxes) if model_fn else self._build_model(input_shape, max_boxes)

    def _build_model(self, img_shape, max_boxes):
        if self.grid_size * self.grid_size * self.boxes_per_grid < max_boxes:
            raise ValueError("Grid resolution too low for max_boxes")

        inputs = tf.keras.Input(shape=img_shape)

        # Block 1 - Start with smaller filters for hyperspectral data
        x = tf.keras.layers.Conv2D(
            16, 3, padding="same", activation="swish", kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.15)(x)  # Increased dropout

        # Block 2
        x = tf.keras.layers.Conv2D(
            32, 3, padding="same", activation="swish", kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        # Block 3
        x = tf.keras.layers.Conv2D(
            64, 3, padding="same", activation="swish", kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.35)(x)

        # Block 4 - Add spatial dropout for better regularization
        x = tf.keras.layers.Conv2D(
            128, 3, padding="same", activation="swish", kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.SpatialDropout2D(0.4)(x)  # Spatial dropout for 2D feature maps

        # Block 5 - Reduced channels to reduce overfitting
        x = tf.keras.layers.Conv2D(
            96,
            3,
            padding="same",
            activation="swish",  # Further reduced from 128
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.SpatialDropout2D(0.5)(x)

        output_channels = self.boxes_per_grid * 5
        x = tf.keras.layers.Conv2D(output_channels, 1, activation="sigmoid")(x)
        x = tf.keras.layers.Reshape((self.grid_size, self.grid_size, self.boxes_per_grid, 5))(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def compile(
        self,
        optimizer=Adam(learning_rate=0.0001),
        loss=yolo_focal_loss(lambda_coord=5.0, lambda_noobj=0.3, alpha=0.25, gamma=2.0),
        metrics=["mae", "accuracy"],
    ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_normalization_constants(self, dataset):
        if self.norm_mean is not None and self.norm_std is not None:
            return tf.constant(self.norm_mean, dtype=tf.float32), tf.constant(self.norm_std, dtype=tf.float32)

        sum_pixels = tf.zeros((dataset.element_spec[0].shape[-1],), dtype=tf.float32)
        sum_squares = tf.zeros((dataset.element_spec[0].shape[-1],), dtype=tf.float32)
        num_pixels = tf.Variable(0, dtype=tf.int32)

        for image, _, _ in dataset:
            num_pixels.assign_add(tf.reduce_prod(tf.shape(image)[:-1]))  # Total pixels across batch
            sum_pixels += tf.reduce_sum(image, axis=[0, 1])  # Sum across height & width
            sum_squares += tf.reduce_sum(tf.square(image), axis=[0, 1])  # Sum of squares

        mean = sum_pixels / tf.cast(num_pixels, tf.float32)
        variance = (sum_squares / tf.cast(num_pixels, tf.float32)) - tf.square(mean)
        stddev = tf.sqrt(tf.maximum(variance, 1e-8))  # Avoid division by zero

        self.norm_mean = tf.constant(mean).numpy()
        self.norm_std = tf.constant(stddev).numpy()

        # last band is not an image band, set to 0 mean, 1 std
        self.norm_mean[-1] = 0.0
        self.norm_std[-1] = 1.0

        return self.norm_mean, self.norm_std

    def preprocess_dataset(self, dataset: tf.data.Dataset):
        def is_valid_sample(image, bboxes):
            """Returns True if the image has at least one valid bounding box."""
            valid_mask = tf.map_fn(has_valid_bbox, bboxes, dtype=tf.bool)
            return tf.reduce_any(valid_mask)

        def is_invalid_sample(image, bboxes):
            """Returns True if the image has only invalid bounding boxes."""
            return tf.logical_not(is_valid_sample(image, bboxes))

        mean, std = self.get_normalization_constants(dataset) if self.normalize else (0.0, 1.0)

        # resize images
        dataset = dataset.map(lambda img, lab, _: (tf.image.resize(img, self.input_shape[:-1]), lab))

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
                [valid_ds, invalid_ds],
                weights=[1 - self.blank_percentage, self.blank_percentage],
                seed=42,
                stop_on_empty_dataset=True,
            )
        else:
            dataset = valid_ds

        dataset = dataset.map(
            lambda img, lab: (img, BBoxModel.convert_boxes_to_grid_labels(lab, self.grid_size, self.boxes_per_grid)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return dataset

    @staticmethod
    def normalize_image(image, mean, std):
        return (image - mean) / std

    @staticmethod
    def convert_boxes_to_grid_labels(boxes, S=16, B=1):
        """
        Convert [max_boxes, 5] boxes to a YOLO-style (S, S, B, 5) grid.
        boxes: [objectness, x_center, y_center, width, height] (all normalized to [0, 1])
        """
        boxes = tf.reshape(boxes, [-1, 5])
        object_mask = boxes[:, 0] > 0.5
        boxes = tf.boolean_mask(boxes, object_mask)

        x, y = boxes[:, 1], boxes[:, 2]
        i = tf.cast(tf.floor(y * S), tf.int32)
        j = tf.cast(tf.floor(x * S), tf.int32)
        i = tf.clip_by_value(i, 0, S - 1)
        j = tf.clip_by_value(j, 0, S - 1)

        # Compute offsets in cell
        x_cell = x * S - tf.cast(j, tf.float32)
        y_cell = y * S - tf.cast(i, tf.float32)

        # Replace the original x, y with offsets
        updated_boxes = tf.stack(
            [
                boxes[:, 0],  # objectness
                x_cell,  # x offset in cell
                y_cell,  # y offset in cell
                boxes[:, 3],  # width
                boxes[:, 4],  # height
            ],
            axis=-1,
        )

        indices = tf.stack([i, j, tf.zeros_like(i)], axis=1)

        _, unique_idx = tf.unique(tf.cast(i * S + j, tf.int32))
        indices = tf.gather(indices, unique_idx)
        updates = tf.gather(updated_boxes, unique_idx)

        grid = tf.tensor_scatter_nd_update(tf.zeros((S, S, B, 5), dtype=tf.float32), indices, updates)
        return grid

    @staticmethod
    def augment_image(image, bboxes, transformation, max_translate=0.2, crop_fraction=0.8):
        """Apply a single transformation to image and bboxes - optimized for hyperspectral methane detection."""
        if transformation == "none":
            return image, bboxes

        # horizontal flip - valid for methane plumes
        if transformation == "horizontal_flip":
            image = tf.image.flip_left_right(image)
            b = tf.identity(bboxes)
            b_y = b[:, 2]
            b_x = 1.0 - b[:, 1]
            return image, tf.stack([b[:, 0], b_x, b_y, b[:, 3], b[:, 4]], axis=1)

        # vertical flip - valid for methane plumes
        if transformation == "vertical_flip":
            image = tf.image.flip_up_down(image)
            b = tf.identity(bboxes)
            b_x = b[:, 1]
            b_y = 1.0 - b[:, 2]
            return image, tf.stack([b[:, 0], b_x, b_y, b[:, 3], b[:, 4]], axis=1)

        # 90-degree rotation CCW - valid for methane plumes
        if transformation == "rotate":
            image = tf.image.rot90(image, k=1)
            b = tf.identity(bboxes)
            x, y, w, h = b[:, 1], b[:, 2], b[:, 3], b[:, 4]
            new_x = y
            new_y = 1.0 - x
            return image, tf.stack([b[:, 0], new_x, new_y, h, w], axis=1)

        # Spectral noise - specific to hyperspectral data
        if transformation == "spectral_noise":
            # Add small amounts of noise across spectral bands
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
            image = tf.add(image, noise)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, bboxes

        # Small random translation - conservative for methane plume detection
        if transformation == "random_translate":
            img_h = tf.shape(image)[0]
            img_w = tf.shape(image)[1]
            dx = tf.random.uniform([], -max_translate, max_translate)
            dy = tf.random.uniform([], -max_translate, max_translate)
            shift_x = tf.cast(dx * tf.cast(img_w, tf.float32), tf.int32)
            shift_y = tf.cast(dy * tf.cast(img_h, tf.float32), tf.int32)
            image = tf.roll(image, shift=[shift_y, shift_x], axis=[0, 1])
            b = tf.identity(bboxes)
            b_x = tf.clip_by_value(b[:, 1] + dx, 0.0, 1.0)
            b_y = tf.clip_by_value(b[:, 2] + dy, 0.0, 1.0)
            return image, tf.stack([b[:, 0], b_x, b_y, b[:, 3], b[:, 4]], axis=1)

        # Conservative random crop - important to preserve plume context
        if transformation == "random_crop":
            img_h = tf.shape(image)[0]
            img_w = tf.shape(image)[1]
            crop_h = tf.cast(crop_fraction * tf.cast(img_h, tf.float32), tf.int32)
            crop_w = tf.cast(crop_fraction * tf.cast(img_w, tf.float32), tf.int32)
            offset_y = tf.random.uniform([], 0, img_h - crop_h + 1, dtype=tf.int32)
            offset_x = tf.random.uniform([], 0, img_w - crop_w + 1, dtype=tf.int32)
            cropped = tf.image.crop_to_bounding_box(image, offset_y, offset_x, crop_h, crop_w)
            image = tf.image.resize(cropped, [img_h, img_w])
            b = tf.identity(bboxes)
            b_x = (b[:, 1] * tf.cast(img_w, tf.float32) - tf.cast(offset_x, tf.float32)) / tf.cast(crop_w, tf.float32)
            b_y = (b[:, 2] * tf.cast(img_h, tf.float32) - tf.cast(offset_y, tf.float32)) / tf.cast(crop_h, tf.float32)
            b_x = tf.clip_by_value(b_x, 0.0, 1.0)
            b_y = tf.clip_by_value(b_y, 0.0, 1.0)
            return image, tf.stack([b[:, 0], b_x, b_y, b[:, 3], b[:, 4]], axis=1)

        # Scale augmentation - slight zoom in/out
        if transformation == "scale":
            scale_factor = tf.random.uniform([], 0.9, 1.1)
            img_h = tf.shape(image)[0]
            img_w = tf.shape(image)[1]
            new_h = tf.cast(tf.cast(img_h, tf.float32) * scale_factor, tf.int32)
            new_w = tf.cast(tf.cast(img_w, tf.float32) * scale_factor, tf.int32)

            # Resize and then crop/pad back to original size
            image = tf.image.resize(image, [new_h, new_w])
            image = tf.image.resize_with_crop_or_pad(image, img_h, img_w)

            # Adjust bounding boxes
            b = tf.identity(bboxes)
            return image, tf.stack([b[:, 0], b[:, 1], b[:, 2], b[:, 3] * scale_factor, b[:, 4] * scale_factor], axis=1)

        raise ValueError(f"Unsupported augmentation: {transformation}")

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
        valid_mask = tf.map_fn(has_valid_bbox, bboxes, dtype=tf.bool)
        valid_count = tf.reduce_sum(tf.cast(valid_mask, tf.float32))
        total_count = tf.cast(tf.shape(bboxes)[0], tf.float32)

        return valid_count / total_count >= threshold

    @staticmethod
    def augment_dataset(image, bbox, augmentations=["none", "horizontal_flip", "vertical_flip"]):
        datasets = []
        for augmentation in augmentations:
            img, box = BBoxModel.augment_image(image, bbox, augmentation)
            datasets.append(tf.data.Dataset.from_tensors((img, box)))

        full_aug_ds = datasets[0]
        for ds in datasets[1:]:
            full_aug_ds = full_aug_ds.concatenate(ds)

        return full_aug_ds

    def train(self, dataset, epochs=10, batch_size=8):
        dataset = dataset.shuffle(1000)
        test_dataset = dataset.enumerate().filter(lambda i, _: i % 4 == 0).map(lambda _, y: y)
        train_dataset = dataset.enumerate().filter(lambda i, _: i % 4 != 0).map(lambda _, y: y)

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=20, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
        ]

        train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
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
            input_shape=model_attrs["input_shape"],
            max_boxes=model_attrs["max_boxes"],
            normalize=model_attrs["normalize"],
            augmentations=model_attrs["augmentations"],
            blank_percentage=model_attrs["blank_percentage"],
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
                "ciou_loss": ciou_loss,
                "loss_fn": yolo_dense_loss(),
            },
            safe_mode=False,  # This has to be false to allow loading the lambda layer
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

    model = BBoxModel(
        (*image_shape, len(IMAGE_FILE_NAMES)),
        max_boxes,
        normalize=normalize,
        augmentations=augmentations,
        blank_percentage=blank_percentage,
    )
    model.compile()
    print(model.model.summary())
    print("Model created successfully.")
    train_dataset = create_bbox_dataset(
        data_dir, max_boxes=max_boxes, exclude_dirs=exclude_dirs, force_square=force_square
    )
    train_dataset = model.preprocess_dataset(train_dataset)
    model.train(train_dataset, epochs=epochs, batch_size=batch_size)

    model.save("./logs")
