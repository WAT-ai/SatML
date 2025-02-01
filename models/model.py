# define/create model
import tensorflow as tf

def create_model(img_shape, max_bbox=10):
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
        
        tf.keras.layers.Dense(4*max_bbox),  

        # reshape output to be [batch_size, num_bboxes, 4]
        tf.keras.layers.Reshape((max_bbox, 4))
    ])
    
    return model
