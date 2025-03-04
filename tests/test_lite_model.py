import tensorflow as tf
import numpy as np

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()  # Allocate memory for model execution

# Get input and output details (i.e. shape of the input tensor, data type required for the input)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input and output details
print("Input Details:", input_details)
print("Output Details:", output_details)

# Create dummy input image with correct shape
input_shape = input_details[0]['shape']
dummy_input = np.random.rand(*input_shape).astype(np.float32)

# Feed the dummy input into the model
interpreter.set_tensor(input_details[0]['index'], dummy_input)

# Run inference
interpreter.invoke()

# Get predicted bounding boxes
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Predicted Bounding Boxes:", output_data)
