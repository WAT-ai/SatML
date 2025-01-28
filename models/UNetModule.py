import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

class Augment(tf.keras.layers.Layer):
  def __init__(self, flip=True, rotate=True, crop=False, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = keras.Sequential()
    self.augment_labels = keras.Sequential()

    if flip:
        self.augment_inputs.add(keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed))
        self.augment_labels.add(keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed))

    if rotate:
        self.augment_inputs.add(keras.layers.RandomRotation(factor=0.1, seed=seed))
        self.augment_labels.add(keras.layers.RandomRotation(factor=0.1, seed=seed))

    if crop:
        self.augment_inputs.add(keras.layers.RandomCrop(height=64, width=64, seed=seed))
        self.augment_inputs.add(keras.layers.Resizing(height=128, width=128))
        self.augment_labels.add(keras.layers.RandomCrop(height=64, width=64, seed=seed))
        self.augment_labels.add(keras.layers.Resizing(height=128, width=128))

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

class UNet():
    """
    Image segmentation module that trains a U-Net model. Adapted from https://www.tensorflow.org/tutorials/images/segmentation.
    """

    def __init__(self, input_channels, num_classes):
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.model = self.unet_model()
        # temporary values for normalization
        self.image_mean = tf.constant([0.10667099, 13.674022, 14.076011, 14.384233,
                                       0.11666037, 1.171873, 0.85495037, 0.4318816,
                                       0.48891786, 0.11440071, 0.11879135, 0.12649247,
                                       0.11130015, 0.10874157, 0.10803088, 0.10755966])
        self.image_std = tf.constant([0.09226333, 13.874699, 15.393596, 17.116693,
                                      0.17859535, 1.4490424, 0.83888125, 0.52144384,
                                      0.6939981, 0.16807413, 0.1728661, 0.17979662,
                                      0.1645278, 0.15060072, 0.138635, 0.12866527])
    
    def load_image(self, datapoint):
        # resize images to 128 x 128 pixels
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(
            datapoint['segmentation_mask'],
            (128, 128),
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )

        # normalize image
        input_image -= self.image_mean
        input_image /= self.image_std

        return input_image, input_mask
    
    def upsample(self, filters, size, apply_dropout=False):
        """Upsamples an input.

        Conv2DTranspose => Batchnorm => Dropout => Relu

        Args:
            filters: number of filters
            size: filter size
            apply_dropout: If True, adds the dropout layer

        Returns:
            Upsample Sequential Model
        """

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result
    
    def unet_model(self):
        """
        Creates a modified U-Net model.

        The encoder/downsampler is a pretrained MobileNetV2 model and will not be trained.

        The decoder/sampler is a series of upsample blocks.
        """

        base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, self.input_channels], include_top=False, weights=None)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        down_stack.trainable = False

        up_stack = [
            self.upsample(512, 3),  # 4x4 -> 8x8
            self.upsample(256, 3),  # 8x8 -> 16x16
            self.upsample(128, 3),  # 16x16 -> 32x32
            self.upsample(64, 3),   # 32x32 -> 64x64
        ]

        inputs = tf.keras.layers.Input(shape=[128, 128, self.input_channels])

        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=self.num_classes, kernel_size=3, strides=2,
            padding='same')  #64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def dice_loss(self, y_true, y_pred):
        # y_true shape [batch_size,128,128,1] dtype=float32
        # y_pred shape [batch_size,128,128,num_classes] dtype=float32
        # y_pred array of logits
        smooth = tf.constant(1e-17)

        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=2, axis=-1)
        y_true = tf.squeeze(y_true)
        y_pred = tf.math.softmax(y_pred, axis=-1)

        numerator = tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        # numerator = tf.cast(numerator, tf.float32)
        # denominator=tf.cast(denominator, tf.float32)
        loss = 1.0 - 2.0 * (numerator + smooth) / (denominator + smooth)

        return loss

    def train(self, train_dataset, val_dataset, epochs=20, batch_size=64, buffer_size=1000, val_subsplits=1, lr=0.001):
       self.model.compile(optimizer='adam',
                        #   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          loss=self.dice_loss,
                          metrics=['accuracy'])
       
       train_images = train_dataset.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
       val_images = val_dataset.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)

       train_batches = (
          train_images
          .cache()
          .shuffle(buffer_size)
          .batch(batch_size)
          .repeat(2)
          .map(Augment())
          .prefetch(buffer_size=tf.data.AUTOTUNE))
       
       val_batches = val_images.batch(batch_size)
       
       model_history = self.model.fit(train_batches, 
                                      epochs=epochs,
                                      validation_data=val_batches)
       
       return model_history

    def create_mask(self, prediction):
        pred_mask = tf.math.softmax(prediction, axis=-1)
        pred_mask = tf.math.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask

    def predict_image(self, image, mask):
        prediction = self.model.predict(image)
        pred_mask = self.create_mask(prediction)
        return (image, mask, pred_mask)

    def predict_dataset(self, test_dataset, batch_size=8):
        test_images = test_dataset.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_batches = test_images.batch(batch_size)
        for images, masks in test_batches.take(1):
            return self.predict_image(images, masks)