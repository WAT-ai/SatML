# purpose: loads trained model and is the program to run.
import tensorflow as tf
import numpy as np
from models.data_preprocessing import stack_images

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

    model = tf.keras.models.load_model("bounding_box_model.keras")
    
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