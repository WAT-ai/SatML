import tensorflow as tf
import rasterio
import numpy as np
from model import create_model, compile_model

model = tf.keras.models.load_model("bounding_box_model.h5")

def process_img(image_path):
    # load the hyperspectral image
    with rasterio.open(image_path) as src:
        # read all bands of the image
        image_data = src.read()
        # convert image data to a tensorflow tensor and reorders to: [height, weight, channels]
        image_tensor = tf.convert_to_tensor(image_data.transpose(1,2,0))
        print("image shape:", image_tensor.shape)
    # return info with a new dimension added at index 0. since tf models expect input in batches, value at index 0 indicates how many images in the batch
    return tf.expand_dims(image_tensor, axis=0)

def predict_bounding_box(image_path):
    processed_img = process_img(image_path)
    # stores the 4 values from the model to predictions
    predicitions = model(processed_img)
    x_left, x_right, y_top, y_bottom = predicitions[0].numpy()
    return x_left, x_right, y_top, y_bottom

if __name__ == "__main__":
    image_path = "/SatML/data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/label_rgba.tif"
    box = predict_bounding_box(image_path)
    print("predicted 4 value coordinates: ", box)

# Get dimensions (assuming the image is in shape [height, width, channels])
# The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns, and channels (if the image is color):
#height, width, channels = image_tensor.shape

# define model that takes in the specific methane-detecting band as input
#model_input_shape = (height, width, channels)
# or model_input_shape = (image_tensor.shape[0], image_tensor.shape[1], methane_gas_channel)



#model.summary()
