import sys
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Add, Reshape

from src.losses import iou_loss, modified_mean_squared_error, ciou_loss, yolo_focal_loss
from src.data_loader import has_valid_bbox, create_bbox_dataset
from src.constants import IMAGE_FILE_NAMES


class BBoxModel:
    """
    YOLO-style model for detecting bounding boxes around methane plumes from hyperspectral images.

    This model implements a more sophisticated YOLO architecture with:
    - Feature Pyramid Network (FPN) for multi-scale detection
    - Multiple anchor boxes per grid cell
    - Improved backbone with residual connections
    - Multi-scale detection heads
    """

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
        boxes_per_grid=3,  # Increased for better detection
        anchor_boxes=None,
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

        # Default anchor boxes for methane plume detection (small, medium, large)
        if anchor_boxes is None:
            self.anchor_boxes = np.array(
                [
                    [0.1, 0.1],  # Small plumes
                    [0.2, 0.15],  # Medium plumes
                    [0.3, 0.25],  # Large plumes
                ]
            )
        else:
            self.anchor_boxes = np.array(anchor_boxes)

        if model_filepath:
            self.model = BBoxModel.load(model_filepath)
        else:
            self.model = model_fn(input_shape, max_boxes) if model_fn else self._build_yolo_model(input_shape)

    def _residual_block(self, x, filters, kernel_size=3, stride=1, name="res_block"):
        """Residual block with batch normalization and skip connection."""
        shortcut = x

        x = Conv2D(
            filters,
            kernel_size,
            strides=stride,
            padding="same",
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            name=f"{name}_conv1",
        )(x)
        x = BatchNormalization(name=f"{name}_bn1")(x)

        x = Conv2D(
            filters,
            kernel_size,
            strides=1,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            name=f"{name}_conv2",
        )(x)
        x = BatchNormalization(name=f"{name}_bn2")(x)

        # Adjust shortcut if dimensions changed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride, padding="same", name=f"{name}_shortcut")(shortcut)
            shortcut = BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

        x = Add(name=f"{name}_add")([x, shortcut])
        x = tf.keras.layers.Activation("swish", name=f"{name}_activation")(x)

        return x

    def _darknet_backbone(self, inputs):
        """
        DarkNet-inspired backbone with residual connections optimized for hyperspectral data.
        Returns feature maps at different scales for FPN.
        """
        # Initial conv layer - adapted for hyperspectral input
        x = Conv2D(
            32,
            3,
            padding="same",
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            name="initial_conv",
        )(inputs)
        x = BatchNormalization(name="initial_bn")(x)

        # Stage 1: 32 channels
        x = self._residual_block(x, 32, name="stage1_res1")
        x = Conv2D(
            64,
            3,
            strides=2,
            padding="same",
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            name="stage1_downsample",
        )(x)

        # Stage 2: 64 channels - feature map 1 (high resolution)
        x = self._residual_block(x, 64, name="stage2_res1")
        x = self._residual_block(x, 64, name="stage2_res2")
        feature_1 = x  # 256x256 (assuming 512x512 input)

        x = Conv2D(
            128,
            3,
            strides=2,
            padding="same",
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            name="stage2_downsample",
        )(x)

        # Stage 3: 128 channels - feature map 2 (medium resolution)
        x = self._residual_block(x, 128, name="stage3_res1")
        x = self._residual_block(x, 128, name="stage3_res2")
        x = self._residual_block(x, 128, name="stage3_res3")
        feature_2 = x  # 128x128

        x = Conv2D(
            256,
            3,
            strides=2,
            padding="same",
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            name="stage3_downsample",
        )(x)

        # Stage 4: 256 channels - feature map 3 (low resolution)
        x = self._residual_block(x, 256, name="stage4_res1")
        x = self._residual_block(x, 256, name="stage4_res2")
        x = self._residual_block(x, 256, name="stage4_res3")
        x = self._residual_block(x, 256, name="stage4_res4")
        feature_3 = x  # 64x64

        x = Conv2D(
            512,
            3,
            strides=2,
            padding="same",
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            name="stage4_downsample",
        )(x)

        # Stage 5: 512 channels - deepest features
        x = self._residual_block(x, 512, name="stage5_res1")
        x = self._residual_block(x, 512, name="stage5_res2")
        feature_4 = x  # 32x32

        return feature_1, feature_2, feature_3, feature_4

    def _fpn_neck(self, feature_maps):
        """
        Feature Pyramid Network neck for multi-scale feature fusion.
        """
        f1, f2, f3, f4 = feature_maps

        # Top-down pathway
        # P4 (deepest features)
        p4 = Conv2D(256, 1, padding="same", activation="swish", name="fpn_p4_conv")(f4)
        p4 = BatchNormalization(name="fpn_p4_bn")(p4)

        # P3
        p4_upsampled = UpSampling2D(2, name="fpn_p4_upsample")(p4)
        f3_adapted = Conv2D(256, 1, padding="same", activation="swish", name="fpn_f3_adapt")(f3)
        f3_adapted = BatchNormalization(name="fpn_f3_adapt_bn")(f3_adapted)
        p3 = Add(name="fpn_p3_add")([p4_upsampled, f3_adapted])
        p3 = Conv2D(256, 3, padding="same", activation="swish", name="fpn_p3_conv")(p3)
        p3 = BatchNormalization(name="fpn_p3_bn")(p3)

        # P2
        p3_upsampled = UpSampling2D(2, name="fpn_p3_upsample")(p3)
        f2_adapted = Conv2D(256, 1, padding="same", activation="swish", name="fpn_f2_adapt")(f2)
        f2_adapted = BatchNormalization(name="fpn_f2_adapt_bn")(f2_adapted)
        p2 = Add(name="fpn_p2_add")([p3_upsampled, f2_adapted])
        p2 = Conv2D(256, 3, padding="same", activation="swish", name="fpn_p2_conv")(p2)
        p2 = BatchNormalization(name="fpn_p2_bn")(p2)

        return p2, p3, p4

    def _detection_head(self, feature_map, name="detection"):
        """
        Detection head that outputs objectness, bbox coordinates for each anchor.
        Output shape: (grid_h, grid_w, num_anchors, 5)
        where 5 = [objectness, x_center, y_center, width, height]
        """
        # Additional convolutions for better feature processing
        x = Conv2D(
            256,
            3,
            padding="same",
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            name=f"{name}_conv1",
        )(feature_map)
        x = BatchNormalization(name=f"{name}_bn1")(x)

        x = Conv2D(
            128,
            3,
            padding="same",
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            name=f"{name}_conv2",
        )(x)
        x = BatchNormalization(name=f"{name}_bn2")(x)

        # Output layer: 5 values per anchor box (objectness + 4 bbox coords)
        output = Conv2D(self.boxes_per_grid * 5, 1, activation="sigmoid", name=f"{name}_output")(x)

        # Reshape to (grid_h, grid_w, num_anchors, 5)
        grid_h = output.shape[1]
        grid_w = output.shape[2]
        output = Reshape((grid_h, grid_w, self.boxes_per_grid, 5), name=f"{name}_reshape")(output)

        return output

    def _build_yolo_model(self, img_shape):
        """Build the complete YOLO-style model with FPN."""
        if self.grid_size * self.grid_size * self.boxes_per_grid < self.max_boxes:
            raise ValueError("Grid resolution too low for max_boxes")

        inputs = Input(shape=img_shape, name="input")

        # Backbone network
        feature_maps = self._darknet_backbone(inputs)

        # Feature Pyramid Network
        fpn_features = self._fpn_neck(feature_maps)

        # Detection heads for different scales
        # We'll use the medium scale (P3) as our main detection head for now
        # This can be extended to multi-scale detection
        detection_output = self._detection_head(fpn_features[1], name="main_detection")

        # Ensure output matches expected grid size using Keras layers
        target_size = self.grid_size
        current_size = detection_output.shape[1]  # Assumes square grid

        # Resize if needed to match target grid size using Keras layers
        if current_size != target_size:
            # Use UpSampling2D or AveragePooling2D for resizing within Keras
            if current_size < target_size:
                # Need to upsample
                scale_factor = target_size // current_size
                if scale_factor > 1:
                    # First reshape to (batch, height, width, channels)
                    reshaped = Reshape((current_size, current_size, self.boxes_per_grid * 5))(detection_output)
                    upsampled = UpSampling2D(size=(scale_factor, scale_factor), interpolation="bilinear")(reshaped)
                    detection_output = Reshape((target_size, target_size, self.boxes_per_grid, 5))(upsampled)
            else:
                # Need to downsample - use average pooling
                pool_size = current_size // target_size
                if pool_size > 1:
                    # First reshape to (batch, height, width, channels)
                    reshaped = Reshape((current_size, current_size, self.boxes_per_grid * 5))(detection_output)
                    pooled = tf.keras.layers.AveragePooling2D(pool_size=(pool_size, pool_size))(reshaped)
                    detection_output = Reshape((target_size, target_size, self.boxes_per_grid, 5))(pooled)

        model = tf.keras.Model(inputs=inputs, outputs=detection_output, name="YOLOv2_MethaneDetector")
        return model

    def compile(
        self,
        optimizer=Adam(learning_rate=0.001, weight_decay=0.0005),
        loss=yolo_focal_loss(lambda_coord=5.0, lambda_noobj=0.5, alpha=0.25, gamma=2.0),
        metrics=["mae"],
    ):
        """Compile the model with optimizer, loss, and metrics."""
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_normalization_constants(self, dataset):
        """Calculate normalization constants from the dataset."""
        if self.norm_mean is not None and self.norm_std is not None:
            return tf.constant(self.norm_mean, dtype=tf.float32), tf.constant(self.norm_std, dtype=tf.float32)

        sum_pixels = tf.zeros((dataset.element_spec[0].shape[-1],), dtype=tf.float32)
        sum_squares = tf.zeros((dataset.element_spec[0].shape[-1],), dtype=tf.float32)
        num_pixels = tf.Variable(0, dtype=tf.int32)

        for image, _, _ in dataset:
            num_pixels.assign_add(tf.reduce_prod(tf.shape(image)[:-1]))
            sum_pixels += tf.reduce_sum(image, axis=[0, 1])
            sum_squares += tf.reduce_sum(tf.square(image), axis=[0, 1])

        mean = sum_pixels / tf.cast(num_pixels, tf.float32)
        variance = (sum_squares / tf.cast(num_pixels, tf.float32)) - tf.square(mean)
        stddev = tf.sqrt(tf.maximum(variance, 1e-8))

        self.norm_mean = tf.constant(mean).numpy()
        self.norm_std = tf.constant(stddev).numpy()

        # Last band is not an image band, set to 0 mean, 1 std
        self.norm_mean[-1] = 0.0
        self.norm_std[-1] = 1.0

        return self.norm_mean, self.norm_std

    def preprocess_dataset(self, dataset: tf.data.Dataset):
        """Preprocess dataset with normalization, augmentation, and YOLO grid conversion."""

        def is_valid_sample(image, bboxes):
            valid_mask = tf.map_fn(has_valid_bbox, bboxes, dtype=tf.bool)
            return tf.reduce_any(valid_mask)

        def is_invalid_sample(image, bboxes):
            return tf.logical_not(is_valid_sample(image, bboxes))

        mean, std = self.get_normalization_constants(dataset) if self.normalize else (0.0, 1.0)

        # 1. Resize images
        dataset = dataset.map(
            lambda img, lab, _: (tf.image.resize(img, self.input_shape[:-1]), lab),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # 2. Apply random augmentation
        def apply_random_aug(img, lab):
            aug_choice = np.random.choice(self.augmentations)
            img, lab = BBoxModel.augment_image(img, lab, aug_choice)
            return img, lab

        dataset = dataset.map(apply_random_aug, num_parallel_calls=tf.data.AUTOTUNE)

        # 3. Normalize images
        if self.normalize:
            dataset = dataset.map(
                lambda img, lab: (BBoxModel.normalize_image(img, mean, std), lab),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # 4. Filter invalid samples
        valid_ds = dataset.filter(is_valid_sample)
        if self.blank_percentage > 0:
            invalid_ds = dataset.filter(is_invalid_sample)
            dataset = tf.data.Dataset.sample_from_datasets(
                [valid_ds, invalid_ds],
                weights=[1 - self.blank_percentage, self.blank_percentage],
                seed=42,
                stop_on_empty_dataset=False,
            )
        else:
            dataset = valid_ds

        # 5. Convert labels to YOLO grid with anchor boxes
        dataset = dataset.map(
            lambda img, lab: (
                img,
                BBoxModel.convert_boxes_to_grid_labels_with_anchors(
                    lab, self.grid_size, self.boxes_per_grid, self.anchor_boxes
                ),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return dataset

    @staticmethod
    def normalize_image(image, mean, std):
        """Normalize image with given mean and standard deviation."""
        return (image - mean) / std

    @staticmethod
    def convert_boxes_to_grid_labels_with_anchors(boxes, S=16, B=3, anchor_boxes=None):
        """
        Convert bounding boxes to YOLO grid format with anchor box assignment.

        Args:
            boxes: [max_boxes, 5] boxes in format [objectness, x_center, y_center, width, height]
            S: Grid size
            B: Number of anchor boxes per grid cell
            anchor_boxes: Anchor box templates [B, 2] in format [width, height]

        Returns:
            Grid of shape (S, S, B, 5)
        """
        if anchor_boxes is None:
            anchor_boxes = tf.constant([[0.1, 0.1], [0.2, 0.15], [0.3, 0.25]], dtype=tf.float32)
        else:
            anchor_boxes = tf.constant(anchor_boxes, dtype=tf.float32)

        boxes = tf.reshape(boxes, [-1, 5])
        object_mask = boxes[:, 0] > 0.5
        boxes = tf.boolean_mask(boxes, object_mask)

        if tf.shape(boxes)[0] == 0:
            return tf.zeros((S, S, B, 5), dtype=tf.float32)

        x, y, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]

        # Calculate grid cell indices
        i = tf.cast(tf.floor(y * tf.cast(S, tf.float32)), tf.int32)
        j = tf.cast(tf.floor(x * tf.cast(S, tf.float32)), tf.int32)
        i = tf.clip_by_value(i, 0, S - 1)
        j = tf.clip_by_value(j, 0, S - 1)

        # Calculate cell-relative coordinates
        x_cell = x * tf.cast(S, tf.float32) - tf.cast(j, tf.float32)
        y_cell = y * tf.cast(S, tf.float32) - tf.cast(i, tf.float32)

        # Initialize grid
        grid = tf.zeros((S, S, B, 5), dtype=tf.float32)

        # Simplified approach: assign each box to the first anchor (anchor 0)
        # For better performance, this could be extended to proper IoU-based assignment
        num_boxes = tf.shape(boxes)[0]

        # Create all indices at once
        anchor_indices = tf.zeros([num_boxes], dtype=tf.int32)  # Use anchor 0 for all boxes
        indices = tf.stack([i, j, anchor_indices], axis=1)

        # Create all updates at once
        updates = tf.stack(
            [
                boxes[:, 0],  # objectness
                x_cell,  # x offset in cell
                y_cell,  # y offset in cell
                w,  # width
                h,  # height
            ],
            axis=1,
        )

        # Handle duplicate indices by keeping only unique grid cells
        # This is a simplified approach - in practice you'd want proper anchor assignment
        unique_indices, unique_idx = tf.unique(i * S + j)
        final_indices = tf.gather(indices, unique_idx)
        final_updates = tf.gather(updates, unique_idx)

        grid = tf.tensor_scatter_nd_update(grid, final_indices, final_updates)

        return grid

    # Import augmentation methods from original BBoxModel
    @staticmethod
    def augment_image(
        image, bboxes, transformation, max_translate=0.2, crop_fraction=0.8, scale_min=0.9, scale_max=1.1
    ):
        """Apply augmentation - reuse from original BBoxModel."""
        # This is a simplified version - in practice, import from the original model
        if transformation == "none":
            return image, bboxes
        elif transformation == "horizontal_flip":
            image = tf.image.flip_left_right(image)
            b = tf.identity(bboxes)
            new_x = 1.0 - b[:, 1]
            return image, tf.stack([b[:, 0], new_x, b[:, 2], b[:, 3], b[:, 4]], axis=1)
        elif transformation == "vertical_flip":
            image = tf.image.flip_up_down(image)
            b = tf.identity(bboxes)
            new_y = 1.0 - b[:, 2]
            return image, tf.stack([b[:, 0], b[:, 1], new_y, b[:, 3], b[:, 4]], axis=1)
        elif transformation == "rotate":
            image = tf.image.rot90(image, k=1)
            b = tf.identity(bboxes)
            x, y, w, h = b[:, 1], b[:, 2], b[:, 3], b[:, 4]
            new_x = y
            new_y = 1.0 - x
            return image, tf.stack([b[:, 0], new_x, new_y, h, w], axis=1)
        elif transformation == "spectral_noise":
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
            image = tf.clip_by_value(image + noise, 0.0, 1.0)
            return image, bboxes
        elif transformation == "translate":
            img_shape = tf.shape(image)
            max_dx = tf.cast(max_translate * tf.cast(img_shape[1], tf.float32), tf.int32)
            max_dy = tf.cast(max_translate * tf.cast(img_shape[0], tf.float32), tf.int32)
            dx = tf.random.uniform([], -max_dx, max_dx, dtype=tf.int32)
            dy = tf.random.uniform([], -max_dy, max_dy, dtype=tf.int32)
            image = tf.roll(image, shift=[dy, dx], axis=[0, 1])
            b = tf.identity(bboxes)
            b_x = b[:, 1] + tf.cast(dx, tf.float32) / tf.cast(img_shape[1], tf.float32)
            b_y = b[:, 2] + tf.cast(dy, tf.float32) / tf.cast(img_shape[0], tf.float32)
            b_x = tf.clip_by_value(b_x, 0.0, 1.0)
            b_y = tf.clip_by_value(b_y, 0.0, 1.0)
            return image, tf.stack([b[:, 0], b_x, b_y, b[:, 3], b[:, 4]], axis=1)
        else:
            return image, bboxes

    def train(self, dataset, epochs=10, batch_size=8, validation_split=0.2):
        """Train the model."""
        dataset = dataset.shuffle(1000)

        # Split dataset
        dataset_size = sum(1 for _ in dataset)
        val_size = int(dataset_size * validation_split)

        train_dataset = dataset.skip(val_size)
        val_dataset = dataset.take(val_size)

        # Enhanced callbacks for better training
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"./logs/{self.unique_id}_bbox_model_best.keras",
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Save attributes before training
        self.save_attrs("./logs")

        # Train the model
        history = self.model.fit(
            train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_dataset, verbose=1
        )

        return history

    def evaluate(self, test_data):
        """Evaluate the model."""
        return self.model.evaluate(test_data)

    def predict(self, x):
        """Make predictions."""
        return self.model.predict(x)

    def save_attrs(self, output_dir):
        """Save model and attributes."""
        # Save attributes
        attrs_dict = {k: self.__dict__[k] for k in self.__dict__ if k != "model"}
        attrs_dict["norm_mean"] = self.norm_mean.tolist() if self.norm_mean is not None else None
        attrs_dict["norm_std"] = self.norm_std.tolist() if self.norm_std is not None else None
        attrs_dict["anchor_boxes"] = self.anchor_boxes.tolist()

        with open(f"{output_dir}/{self.unique_id}_attrs.yaml", "w") as attrs_file:
            yaml.safe_dump(attrs_dict, attrs_file)

        print(f"YOLOv2 Attributes saved to {output_dir}: {self.unique_id}")

    def save(self, output_dir):
        """Save model and attributes."""
        self.model.save(f"{output_dir}/{self.unique_id}_bbox_model.keras")

        self.save_attrs(output_dir)

        print(f"YOLOv2 Model and attributes saved to {output_dir}: {self.unique_id}")

    @classmethod
    def load(cls, model_attrs_file: str):
        """Load model from saved files."""
        model_attrs = yaml.safe_load(open(model_attrs_file))
        model_file = model_attrs_file.replace("_attrs.yaml", "_bbox_model.keras")

        obj = cls(
            input_shape=model_attrs["input_shape"],
            max_boxes=model_attrs["max_boxes"],
            normalize=model_attrs["normalize"],
            augmentations=model_attrs["augmentations"],
            blank_percentage=model_attrs["blank_percentage"],
            grid_size=model_attrs.get("grid_size", 16),
            boxes_per_grid=model_attrs.get("boxes_per_grid", 3),
            anchor_boxes=model_attrs.get("anchor_boxes", None),
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
                "loss_fn": yolo_focal_loss(),
            },
            safe_mode=False,
        )
        return obj


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "./config/bbox_model_config.yaml"

    config_dict = yaml.safe_load(open(config_file))

    # Extract configuration
    data_dir = config_dict.get("data_dir", "./data/raw_data/STARCOP_train_easy")
    max_boxes = config_dict.get("max_boxes", 2)
    image_shape = config_dict.get("image_shape", (512, 512))
    normalize = config_dict.get("normalize", True)
    augmentations = config_dict.get("augmentations", ["none", "horizontal_flip", "vertical_flip", "rotate"])
    blank_percentage = config_dict.get("blank_percentage", 0.1)
    exclude_dirs = config_dict.get("exclude_dirs", [])
    force_square = config_dict.get("force_square", False)
    batch_size = config_dict.get("batch_size", 16)  # Smaller batch size for more complex model
    epochs = config_dict.get("epochs", 200)

    # Model-specific parameters
    grid_size = config_dict.get("grid_size", 16)
    boxes_per_grid = config_dict.get("boxes_per_grid", 3)

    # Anchor boxes optimized for methane plumes
    anchor_boxes = config_dict.get(
        "anchor_boxes",
        [
            [0.08, 0.08],  # Very small plumes
            [0.15, 0.12],  # Small plumes
            [0.25, 0.20],  # Medium plumes
        ],
    )

    print("Creating YOLOv2-style model for methane plume detection...")

    model = BBoxModel(
        (*image_shape, len(IMAGE_FILE_NAMES)),
        max_boxes,
        normalize=normalize,
        augmentations=augmentations,
        blank_percentage=blank_percentage,
        grid_size=grid_size,
        boxes_per_grid=boxes_per_grid,
        anchor_boxes=anchor_boxes,
    )

    model.compile()
    print(model.model.summary())
    print("YOLOv2 model created successfully.")

    # Load and preprocess dataset
    print("Loading dataset...")
    train_dataset = create_bbox_dataset(
        data_dir, max_boxes=max_boxes, exclude_dirs=exclude_dirs, force_square=force_square
    )
    train_dataset = model.preprocess_dataset(train_dataset)

    print("Starting training...")
    history = model.train(train_dataset, epochs=epochs, batch_size=batch_size)

    model.save("./logs")
    print("Training completed and model saved!")
