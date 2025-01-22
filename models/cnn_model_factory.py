from tensorflow import keras
from src.losses import weighted_bce_plus_dice

def get_no_downsample_cnn_model(
    input_shape, output_channels, loss_weight, show_summary=False
):
    """
    Simple CNN model for hyperspectral segmentation.

    Args:
        input_shape: (H, W, C), the input shape of hyperspectral images.
        output_channels: Number of output channels (usually 1 for binary segmentation).

    Returns:
        A compiled Keras model.
    """
    if output_channels != 1:
        raise NotImplementedError("Only binary segmentation is supported.")

    def se_block(x, reduction=4):
        filters = x.shape[-1]
        se = keras.layers.GlobalAveragePooling2D()(x)
        se = keras.layers.Dense(filters // reduction, activation="relu")(se)
        se = keras.layers.Dense(filters, activation="sigmoid")(se)
        return keras.layers.multiply([x, se])

    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (1, 1), activation="swish", padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = se_block(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="swish", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = se_block(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="swish", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = se_block(x)

    x = keras.layers.Conv2D(32, (3, 3), activation="swish", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = se_block(x)

    # Output layer: 1x1 convolution to predict the binary segmentation mask
    outputs = keras.layers.Conv2D(
        output_channels, (1, 1), activation="sigmoid", padding="same"
    )(x)

    # Build and compile the model
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_bce_plus_dice(loss_weight),
        # loss=dice_loss,
        metrics=["accuracy"],
    )

    if show_summary:
        model.summary()
    return model
