# purpose: loads trained model and is the program to run.
import tensorflow as tf
import numpy as np
from .data_preprocessing import stack_images

model = tf.keras.models.load_model("bounding_box_model.keras")

def predict_bounding_boxes(model, image):
    """
    Predicts bounding box coordinates for the given image
    Parameters:
        model: the tf model.
        image (np.ndarray): the preprocessed image data (single image).
    Returns:
        list: Predicted bounding box coordinates [x_min, y_min, x_max, y_max].
    """
    # expand dimensions to match the model input shape (batch size of 1)
    image = np.expand_dims(image, axis=0)
    
    predicted_coordinates = model.predict(image)
    
    # model output is [x_min, y_min, x_max, y_max]
    return predicted_coordinates[0].tolist()

if __name__ == "__main__":
    
    image_paths = [
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_AVIRIS_460nm.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_AVIRIS_550nm.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_AVIRIS_640nm.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_AVIRIS_2004nm.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_AVIRIS_2109nm.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_AVIRIS_2310nm.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_AVIRIS_2350nm.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_AVIRIS_2360nm.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_WV3_SWIR1.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_WV3_SWIR2.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_WV3_SWIR3.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_WV3_SWIR4.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_WV3_SWIR5.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_WV3_SWIR6.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_WV3_SWIR7.tif',
        './data/raw_data/STARCOP_train_easy/ang20190922t192642_r4578_c217_w151_h151/TOA_WV3_SWIR8.tif'
    ] 
    # loads and preprocesses the images
    image = stack_images(image_paths) 

    # predict bounding boxes
    bounding_boxes = predict_bounding_boxes(model, image)
    
    print("predicted bounding box coordinates:", bounding_boxes)

#ignore below (old code)








# def predict_bounding_box(model, image_path):
#     processed_img = process_img(image_path)
#     # stores 4 values from the model to predictions
#     predicitions = model(processed_img)
#     x_left, x_right, y_top, y_bottom = predicitions[0].numpy()
#     return x_left, x_right, y_top, y_bottom

# if __name__ == "__main__":
#     image_path = "./data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_AVIRIS_640nm.tif"
#     box = predict_bounding_box(image_path)
#     print("predicted 4 value coordinates: ", box)

# def process_img(image_path):
#     # load the hyperspectral image
#     with rasterio.open(image_path) as src:
#         # read all bands of the image
#         image_data = src.read()
#         print("Metadata:", src.meta)

#         # read all bands explicitly
#         image_data = src.read(tuple(range(1, src.count + 1)))
#         print("Loaded image data shape:", image_data.shape)
        
#         # convert image data to a tensorflow tensor and reorders to: [height, weight, channels]
#         image_tensor = tf.convert_to_tensor(image_data.transpose(1,2,0))
#         print("image shape:", image_tensor.shape)

#     # return info with a new dimension added at index 0. since tf models expect input in batches, value at index 0 indicates how many images in the batch
#     return tf.expand_dims(image_tensor, axis=0)

#model.summary()