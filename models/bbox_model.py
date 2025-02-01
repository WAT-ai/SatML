import tensorflow as tf
from tensorflow.keras.models import load_model

from src.image_utils import compare_bbox
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

        if model_filepath:
            self.model = BBoxModel.load(model_filepath)
        else:
            self.model = model_fn(input_shape, max_boxes) if model_fn else self._build_model(input_shape, max_boxes)

    def _build_model(self, img_shape, max_boxes):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=img_shape),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(4 * max_boxes),
                # reshape output to be [batch_size, num_bboxes, 4]
                tf.keras.layers.Reshape((max_boxes, 4)),
            ]
        )
        return model

    def compile(
        self,
        optimizer="adam",
        loss=compare_bbox,
        metrics=["mae", "accuracy"],
    ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_normalization_constants(self, dataset):
        if self.norm_mean is not None and self.norm_std is not None:
            return tf.constant(self.norm_mean, dtype=tf.float32), tf.constant(self.norm_std, dtype=tf.float32)

        sum_pixels = tf.zeros((16,), dtype=tf.float32)
        sum_squares = tf.zeros((16,), dtype=tf.float32)
        num_pixels = tf.Variable(0, dtype=tf.int32)

        for image_batch, _ in dataset:
            num_pixels.assign_add(tf.reduce_prod(tf.shape(image_batch)[:-1]))  # Total pixels across batch
            sum_pixels += tf.reduce_sum(image_batch, axis=[0, 1, 2])  # Sum across height & width
            sum_squares += tf.reduce_sum(tf.square(image_batch), axis=[0, 1, 2])  # Sum of squares

        mean = sum_pixels / tf.cast(num_pixels, tf.float32)
        variance = (sum_squares / tf.cast(num_pixels, tf.float32)) - tf.square(mean)
        stddev = tf.sqrt(variance)

        self.norm_mean = tf.constant(mean).numpy()
        self.norm_std = tf.constant(stddev).numpy()

        print(f"Normalization constants: mean={self.norm_mean}, stddev={self.norm_std}")

        return mean, stddev

    def preprocess_dataset(self, dataset: tf.data.Dataset):
        mean, std = self.get_normalization_constants(dataset) if self.normalize else (0.0, 1.0)

        # normalize images
        if self.normalize:
            dataset = dataset.map(lambda img, lab: BBoxModel.normalize_image(img, mean, std))

        # augment images
        dataset = dataset.flat_map(lambda img, label: BBoxModel.augment_dataset(img, label, self.augmentations))

        return dataset

    @staticmethod
    def normalize_image(image, mean, std):
        return (image - mean) / std

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
        train_dataset = train_dataset.batch(batch_size).prefetch(1)
        return self.model.fit(train_dataset, epochs=epochs)

    def evaluate(self, test_data):
        return self.model.evaluate(test_data)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filepath):
        self.model.save(filepath)

    @staticmethod
    def load(filepath):
        model = load_model(filepath, custom_objects={"compare_bbox": compare_bbox})
        return model


if __name__ == "__main__":
    data_dir = './data/raw_data/STARCOP_train_easy'
    model = BBoxModel((512, 512, 16), 1, normalize=False, augmentations=["none", "horizontal_flip"])
    model.compile()
    print(model.model.summary())
    print("Model created successfully.")
    train_dataset = create_bbox_dataset(data_dir, max_boxes=1)
    train_dataset = model.preprocess_dataset(train_dataset)
    model.train(train_dataset, epochs=1)
