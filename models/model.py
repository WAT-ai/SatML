# define/create model
import tensorflow as tf

def create_model(img_shape):
    model = tf.keras.Sequential([
        # conv layer captures spatial-spectral features
        # (4,4) is the kernal size of the filter that slides over the input image
        # 64 filters which allows to capture moer complex patterns
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape = img_shape),
        
        # reduce computational load, lot of pixels => large number of computations in each layer of the network, less memory needed
        tf.keras.layers.MaxPooling2D((2, 2)),

        # extracts features from the layer
        # detects patterns in the data
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(4)
    ])
    return model