
import rasterio
import tensorflow as tf

# load the hyperspectral image
with rasterio.open(image_path) as src:
    # read all bands of the image
    image_data = src.read()
    # [channels, height, width] is returned

# convert image data to a tensorflow tensor and reorders to: [height, weight, channels]
image_tensor = tf.convert_to_tensor(image_data.transpose(1,2,0))

print("image shape:", image_tensor.shape)

# Get dimensions (assuming the image is in shape [height, width, channels])
# The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns, and channels (if the image is color):
height, width, channels = image_tensor.shape

# do i need to know how to access a specific wavelength?
# should i be finding the channel that corresponds to the wavelength that can detect methan emissions?
# methane_gas_channel = image_tensor[:,:,10];

# define model that takes in the specific methane-detecting band as input
model_input_shape = (height, width, channels)
# or model_input_shape = (image_tensor.shape[0], image_tensor.shape[1], methane_gas_channel)

model = tf.keras.Sequential([
    # First Conv layer to capture spatial-spectral features
    # (4,4) is the kernal size of the filter that slides over the input image
    # 64 filters which allows to capture moer complex patterns
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=model_input_shape),
    # reduce computational load, lot of pixels => large number of computations in each layer of the network, less memory needed
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten and fully connected layers for coordinate regression
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),

    # output layer with 4 units for x_left, x_right, y_top, y_bottom
    tf.keras.layers.Dense(4) 
])

model.summary()
