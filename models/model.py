# define, compile, and train
import rasterio
import tensorflow as tf

def load_img(image_path):
    # load the hyperspectral image
    with rasterio.open(image_path) as src:
    # read all bands of the image
        image_data = src.read()
        # convert image data to a tensorflow tensor and reorders to: [height, weight, channels]
        image_tensor = tf.convert_to_tensor(image_data.transpose(1,2,0))
        print("image shape:", image_tensor.shape)
        # [channels, height, width] is returned
    return image_data.transpose(1,2,0)

def create_model(img_shape):
    model = tf.keras.Sequential([
        # conv layer to capture spatial-spectral features
        # (4,4) is the kernal size of the filter that slides over the input image
        # 64 filters which allows to capture moer complex patterns
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', img_shape=img_shape),
        # reduce computational load, lot of pixels => large number of computations in each layer of the network, less memory needed
        tf.keras.layers.MaxPooling2D((2, 2)),

        # extracts features from the layer
        # detects patterns in the data
        #  
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),

        # final layer with x_left, x_right, y_top, y_bottom values
        tf.keras.layers.Dense(4)
        # layers are for predicting the bounding box coordinates 
    ])
    return model

# compiling the model
# MSE penalizes large errors which in return minimizies coordinate errors
# MAE provides a more direct measure of average error
def compile_model(model):
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# training the model
def train_model(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    model.save("bounding_box_model.h5")  
    return model

if __name__ == "__main__":
    # load the image to get dimensions
    image_path = "/SatML/data/raw_data/STARCOP_train_easy/ang20190922t192642_r2048_c0_w512_h512/label_rgba.tif"
    image_tensor = load_img(image_path)
    
    # stores the info in img_shape
    img_shape = image_tensor.shape

    # call the funcs to create and compile the model
    model = create_model(img_shape)
    model = compile_model(model)