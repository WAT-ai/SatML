import tensorflow as tf

from src.losses import iou_loss, modified_mean_squared_error


if __name__ == "__main__":

    try:
        model = tf.keras.models.load_model(
            "bbox_model.keras", 
            custom_objects={"iou_loss": iou_loss, 
            "modified_mean_squared_error": modified_mean_squared_error}
        )
        model.summary()

        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.allow_custom_ops = True
        tflite_model = converter.convert()

        # Saving the model
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("Model successfully converted to TensorFlow Lite.")
    except Exception as e:
        print("Error during conversion:", e)
        raise
