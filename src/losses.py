import tensorflow as tf
import tensorflow.keras.backend as K


def dice_loss(y_true, y_pred, smooth=1):
    """Calculates the dice loss (negative dice coefficient)

    Args:
        y_true (tf.tensor): GT masks
        y_pred (tf.tensor): predicted masks
        smooth (int, optional): smoothing coefficient for dice. Defaults to 1.

    Returns:
        tf.tensor: dice coefficient
    """
    y_pred_classes = y_pred.shape[-1]
    y_true_classes = y_true.shape[-1]

    if y_pred_classes != y_pred_classes and y_true_classes != 1:
        raise ValueError(f"Number of classes in GT and predicted masks do not match. {y_true_classes} != {y_pred_classes}")

    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=y_pred_classes) if y_pred_classes > 1 else tf.cast(y_true, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -((2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def weighted_dice_loss(y_true, y_pred, smooth=1, weight=5.0):
    """Calculates the weighted dice loss using given weight (negative dice coefficient)

    Args:
        y_true (tf.tensor): GT masks
        y_pred (tf.tensor): predicted masks
        smooth (int, optional): smoothing coefficient for dice. Defaults to 1.
        weight (float, optional): weight of positive class. Defaults to 5.0.

    Returns:
        tf.tensor: weighted dice coefficient
    """
    y_pred_classes = y_pred.shape[-1]
    y_true_classes = y_true.shape[-1]

    if y_pred_classes != y_pred_classes and y_true_classes != 1:
        raise ValueError(f"Number of classes in GT and predicted masks do not match. {y_true_classes} != {y_pred_classes}")

    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=y_pred_classes) if y_pred_classes > 1 else tf.cast(y_true, tf.float32)

    # Flatten tensors
    y_true_f = tf.reshape(y_true, shape=(-1,))
    y_pred_f = tf.reshape(y_pred, shape=(-1,))

    # Separate weights for classes
    intersection = tf.reduce_sum(weight * y_true_f * y_pred_f)
    dice_coeff = (2.0 * intersection + smooth) / (weight *
                                                  tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return -dice_coeff


def weighted_bce_loss(y_true, y_pred, weight=5.0):
    """Calculates the weighted binary crossentropy

    Args:
        y_true (tf.tensor): GT masks
        y_pred (tf.tensor): predicted masks
        weight (float, optional): weight of positive class. Defaults to 5.0.

    Returns:
        tf.tensor: weighted binary crossentropy value
    """
    bce = K.binary_crossentropy(y_true, y_pred)
    weight_map = (y_true * weight) + 1
    return weight_map * bce


def weighted_bce_plus_dice(weight: float):
    """Wrapper function for weighted binary crossentropy + Dice loss
    NOTE: Dice doesn't use weights here

    Args:
        weight (float): weight of positive class. Defaults to 5.0.
    """
    def _weighted_bce_plus_dice(y_true, y_pred):
        dice = dice_loss(y_true, y_pred)
        bce = weighted_bce_loss(
            y_true, y_pred, tf.constant(weight, tf.float32))

        return dice + bce
    return _weighted_bce_plus_dice


def focal_loss(alpha=0.25, gamma=2):
    """Wrapper function for focal loss

    NOTE: you could also use keras.losses.BinaryFocalCrossentropy

    Args:
        alpha (float, optional): balances importance of positive and negative samples. Defaults to 0.25.
        gamma (int, optional): controls the shape of loss function.
            0 equates to BCE. Higher values mean more attention towards harder to classify examples
            Defaults to 2.
    """
    def _focal_loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype=y_pred.dtype)
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = alpha * K.pow(1 - p_t, gamma) * bce
        return fl

    return _focal_loss
