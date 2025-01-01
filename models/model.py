# define/create model
import tensorflow as tf

def create_model(img_shape, num_of_boxes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=img_shape),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        
        tf.keras.layers.Dense(4*num_of_boxes),  # output will look like: [x_min, y_min, x_max, y_max]

        # reshape output to be [batch_size, num_bboxes, 4]
        tf.keras.layers.Reshape((num_of_boxes, 4))
    ])
    
    return model
