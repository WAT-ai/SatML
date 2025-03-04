import tensorflow as tf

from tensorflow.keras.models import load_model
from src.losses import iou_loss, modified_mean_squared_error

# Convert a Keras model into a TensorFlow Lite model
model = tf.keras.models.load_model("bbox_model.h5", custom_objects={"iou_loss": iou_loss, "modified_mean_squared_error": modified_mean_squared_error})


# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
