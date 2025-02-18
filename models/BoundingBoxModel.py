import yaml
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from models import BaseModel
from src.losses import iou_loss, modified_mean_squared_error

class BoundingBoxModel(BaseModel):

    def __init__(self, input_shape, max_boxes, model_fn=None, model_filepath=None):
        super(BoundingBoxModel, self).__init__()
        self.input_shape = input_shape
        self.max_boxes = max_boxes
        self.unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

        if model_filepath:
            self.model = BoundingBoxModel.load(model_filepath)
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
        loss=modified_mean_squared_error,
        metrics=["mae", "accuracy"],
    ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

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

        with open(f"{output_dir}/{self.unique_id}_attrs.yaml", "w") as attrs_file:
            yaml.safe_dump(attrs_dict, attrs_file)

    @staticmethod
    def load(filepath):
        model = load_model(filepath, custom_objects={"iou_loss": iou_loss, "modified_mean_squared_error": modified_mean_squared_error})
        return model