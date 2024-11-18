# training the model
from .model import create_model
from .data_preprocessing import stack_images, preprocess_images, augment_data
from keras.src.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np

image_paths = [
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_AVIRIS_460nm.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_AVIRIS_550nm.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_AVIRIS_640nm.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_AVIRIS_2004nm.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_AVIRIS_2109nm.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_AVIRIS_2310nm.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_AVIRIS_2350nm.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_AVIRIS_2360nm.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_WV3_SWIR1.tif', 
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_WV3_SWIR2.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_WV3_SWIR3.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_WV3_SWIR4.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_WV3_SWIR5.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_WV3_SWIR6.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_WV3_SWIR7.tif',
    './data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/TOA_WV3_SWIR8.tif'
]

# CAN DELETE
annotations = [ # [x_min, y_min, x_max, y_max]
    [50, 60, 200, 220],  
    [100, 120, 250, 270],  
    [20, 120, 250, 270],
    [100, 90, 300, 400],
    [35, 28, 100, 100],
    [150, 5, 222, 80],
    [80, 100, 200, 300],
    [92, 55, 400, 180],
    [50, 60, 200, 220],  
    [100, 120, 250, 270],  
    [20, 120, 250, 270],
    [100, 90, 300, 400],
    [35, 28, 100, 100],
    [150, 5, 222, 80],
    [80, 100, 200, 300],
    [92, 55, 400, 180]
]

def train_model(images, annotations):
    """
    load data, preprocess it, and train the model
    Parameters:
        images: "Directory" containing training images
        annotations: "Directory" containing annotations
    """

    # stack the images using the function in data_preprocessing
    images = stack_images(image_paths)

    img_shape = images.shape

    # (NOT USING RIGHT NOW) preprocess images i.e resize
    # preprocessed_image = preprocess_images(images)

    # (NOT USING RIGHT NOW) augment data
    # augmented_image = augment_data(preprocessed_image)

    model = create_model(img_shape)  

    # checkpoint saves the best model during training
    # checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

    # compiling the model
    # MSE penalizes large errors which in return minimizies coordinate errors
    # MAE provides a more direct measure of average error
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

    training_images = [images]
    training_images = np.array(training_images)
    
    training_annotations = [[20, 120, 250, 270]]
    training_annotations = np.array(training_annotations)

    # training the model
    model.fit(
        training_images, 
        training_annotations, 
        epochs=10, 
        batch_size=32, 
        # validation_split=0.2
        # callbacks=[checkpoint]
    )

    model.save('bounding_box_model.keras')
    print("bounding box model saved successfully")

if __name__ == "__main__":
    train_model(image_paths, annotations)
