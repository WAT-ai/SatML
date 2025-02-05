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

def modified_mean_squared_error(y_true, y_pred):
    # Reshape from (batch, n, 4) to (batch * n, 4)
    y_true = tf.reshape(y_true, [-1, 4])
    y_pred = tf.reshape(y_pred, [-1, 4])

    # Check if the ground truth and predicted boxes are all negative (invalid bounding box)
    all_negative_true = tf.reduce_all(y_true < 0, axis=1)
    all_negative_pred = tf.reduce_all(y_pred < 0, axis=1)
    
    # Mask where both boxes are negative
    valid_boxes_mask = tf.logical_and(all_negative_true, all_negative_pred)
    
    # Set loss to 0 for invalid boxes, keep normal loss for valid boxes
    final_loss = tf.where(valid_boxes_mask, 0.0, tf.reduce_mean(tf.square(y_true - y_pred), axis=1))

    return tf.reduce_mean(final_loss)  # Reduce over batch

def iou_loss(y_true, y_pred):
    """
    Modified Iou loss function for bounding box regression.
    
    Args:
        y_true: Tensor of shape (batch_size, n, 4) with ground truth bounding boxes.
        y_pred: Tensor of shape (batch_size, n, 4) with predicted bounding boxes.
    
    Returns:
        A scalar tensor representing the loss.
    """

    # Mask for valid bounding boxes
    valid_mask = tf.reduce_all(y_true >= 0, axis=-1, keepdims=True)  # Shape: (batch_size, n, 1)
    pred_valid_mask = tf.reduce_all(y_pred >= 0, axis=-1, keepdims=True)  # (batch_size, n, 1)

    # Extract box coordinates
    x1_true, x2_true, y1_true, y2_true = tf.split(y_true, 4, axis=-1)
    x1_pred, x2_pred, y1_pred, y2_pred = tf.split(y_pred, 4, axis=-1)

    # Compute Intersection
    x1_int = tf.maximum(x1_true, x1_pred)
    x2_int = tf.minimum(x2_true, x2_pred)
    y1_int = tf.maximum(y1_true, y1_pred)
    y2_int = tf.minimum(y2_true, y2_pred)

    inter_area = tf.maximum(0.0, x2_int - x1_int) * tf.maximum(0.0, y2_int - y1_int)

    # Compute Union
    area_true = (x2_true - x1_true) * (y2_true - y1_true)
    area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union_area = area_true + area_pred - inter_area

    # IoU Calculation (avoid division by zero)
    iou = inter_area / (union_area + 1e-6)
    iou_loss = 1.0 - iou  # 1 - IoU to make it a loss

    # L1 Loss for precise localization
    l1_loss = tf.abs(y_true - y_pred)

    # Compute regular loss only for valid boxes
    reg_loss = (iou_loss + 0.5 * l1_loss) * tf.cast(valid_mask, tf.float32)

    # Penalize cases where y_true is invalid but y_pred is valid
    invalid_pred_penalty = tf.cast(~valid_mask & pred_valid_mask, tf.float32) * 10.0

    # Total loss
    total_loss = reg_loss + invalid_pred_penalty

    # Compute final loss by averaging over valid cases
    return tf.reduce_sum(total_loss) / (tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + tf.reduce_sum(invalid_pred_penalty) + 1e-6)
