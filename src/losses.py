import tensorflow as tf
import tensorflow.keras.backend as K


def yolo_dense_loss(lambda_box=5.0, lambda_obj=1.0):
    """
    Custom YOLO-lite loss for Dense-based output (no classification, 1 class).

    Parameters:
        lambda_box: weight for the box regression loss
        lambda_obj: weight for the objectness loss

    Returns:
        A TensorFlow loss function that takes (y_true, y_pred)
    """

    def loss_fn(y_true, y_pred):
        # Split objectness and box coords
        obj_true = y_true[..., 0]  # shape (batch, max_boxes)
        box_true = y_true[..., 1:]  # shape (batch, max_boxes, 4)

        obj_pred = y_pred[..., 0]
        box_pred = y_pred[..., 1:]

        # Objectness loss (BCE)
        obj_loss = tf.keras.backend.binary_crossentropy(obj_true, obj_pred)

        if obj_loss.shape != obj_true.shape:
            raise ValueError(f"Objectness loss shape mismatch {obj_loss.shape} != {obj_true.shape}")

        # Box loss (L2), only where objectness == 1
        box_loss = tf.reduce_sum(tf.square(box_true - box_pred), axis=-1)  # (batch, max_boxes)
        box_loss = tf.where(obj_true > 0.0, box_loss, tf.zeros_like(box_loss))

        # Final combined loss
        total_loss = lambda_obj * obj_loss + lambda_box * box_loss
        return tf.reduce_mean(total_loss)

    return loss_fn


def yolo_grid_loss(lambda_coord=5.0, lambda_noobj=0.5):
    def loss_fn(y_true, y_pred):
        # Shape: (batch, S, S, B, 5)
        object_mask = y_true[..., 0]

        # Coordinate loss (only where object exists)
        xy_loss = tf.reduce_sum(tf.square(y_true[..., 1:3] - y_pred[..., 1:3]) * object_mask[..., tf.newaxis])

        wh_loss = tf.reduce_sum(tf.square(y_true[..., 3:5] - y_pred[..., 3:5]) * object_mask[..., tf.newaxis])

        # Objectness loss (binary cross-entropy)
        bce = tf.keras.backend.binary_crossentropy
        obj_loss = tf.reduce_sum(bce(y_true[..., 0], y_pred[..., 0]))

        # No-object loss (optional — discourage false positives)
        noobj_mask = 1.0 - object_mask
        noobj_loss = tf.reduce_sum(lambda_noobj * bce(tf.zeros_like(y_pred[..., 0]), y_pred[..., 0]) * noobj_mask)

        total_loss = lambda_coord * (xy_loss + wh_loss) + obj_loss + noobj_loss

        return total_loss / tf.cast(tf.shape(y_true)[0], tf.float32)  # average over batch

    return loss_fn


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
        raise ValueError(
            f"Number of classes in GT and predicted masks do not match. {y_true_classes} != {y_pred_classes}"
        )

    y_true = (
        tf.one_hot(tf.cast(y_true, tf.uint8), depth=y_pred_classes)
        if y_pred_classes > 1
        else tf.cast(y_true, tf.float32)
    )

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
        raise ValueError(
            f"Number of classes in GT and predicted masks do not match. {y_true_classes} != {y_pred_classes}"
        )

    y_true = (
        tf.one_hot(tf.cast(y_true, tf.uint8), depth=y_pred_classes)
        if y_pred_classes > 1
        else tf.cast(y_true, tf.float32)
    )

    # Flatten tensors
    y_true_f = tf.reshape(y_true, shape=(-1,))
    y_pred_f = tf.reshape(y_pred, shape=(-1,))

    # Separate weights for classes
    intersection = tf.reduce_sum(weight * y_true_f * y_pred_f)
    dice_coeff = (2.0 * intersection + smooth) / (weight * tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
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
        bce = weighted_bce_loss(y_true, y_pred, tf.constant(weight, tf.float32))

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
    # IoU Loss
    y_true_reshaped = tf.reshape(y_true, [-1, 4])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 4])

    valid_mask = tf.reduce_any(y_true_reshaped > 0, axis=-1, keepdims=True)
    x1_true, x2_true, y1_true, y2_true = tf.split(y_true_reshaped, 4, axis=-1)
    x1_pred, x2_pred, y1_pred, y2_pred = tf.split(y_pred_reshaped, 4, axis=-1)

    x1_int = tf.maximum(x1_true, x1_pred)
    x2_int = tf.minimum(x2_true, x2_pred)
    y1_int = tf.maximum(y1_true, y1_pred)
    y2_int = tf.minimum(y2_true, y2_pred)

    inter_area = tf.maximum(0.0, x2_int - x1_int) * tf.maximum(0.0, y2_int - y1_int)
    area_true = tf.maximum(0.0, (x2_true - x1_true) * (y2_true - y1_true))
    area_pred = tf.maximum(0.0, (x2_pred - x1_pred) * (y2_pred - y1_pred))
    union_area = area_true + area_pred - inter_area + 1e-6

    iou = inter_area / union_area
    # valid_mask = tf.cast(valid_mask, tf.bool)
    iou_loss = (1 - iou) * tf.cast(valid_mask, tf.float32)

    # Smooth L1 Loss for coordinate regression
    smooth_l1 = tf.losses.huber(y_true_reshaped, y_pred_reshaped)

    # Weighted Combination
    total_loss = tf.reduce_mean(iou_loss) + 0.5 * smooth_l1
    return total_loss


def ciou_loss(y_true, y_pred, lambda_reg=10, img_size=512):
    # Extract coordinates
    x_left_true, x_right_true, y_top_true, y_bottom_true = tf.split(y_true * img_size, 4, axis=-1)
    x_left_pred, x_right_pred, y_top_pred, y_bottom_pred = tf.split(y_pred * img_size, 4, axis=-1)

    # Validate boxes
    valid_pred_mask = (x_left_pred < x_right_pred) & (y_top_pred < y_bottom_pred)
    valid_true_mask = (x_left_true < x_right_true) & (y_top_true < y_bottom_true)
    valid_mask = tf.cast(valid_pred_mask & valid_true_mask, tf.float32)

    # Penalty for invalid predicted boxes
    invalid_penalty = tf.cast(~valid_pred_mask, tf.float32) * 1.0  # Penalty factor

    # Ensure coordinates are valid
    x_left_pred = tf.minimum(x_left_pred, x_right_pred)
    y_top_pred = tf.minimum(y_top_pred, y_bottom_pred)
    x_left_true = tf.minimum(x_left_true, x_right_true)
    y_top_true = tf.minimum(y_top_true, y_bottom_true)

    # Areas
    area_true = tf.maximum(0.0, (x_right_true - x_left_true) * (y_bottom_true - y_top_true))
    area_pred = tf.maximum(0.0, (x_right_pred - x_left_pred) * (y_bottom_pred - y_top_pred))

    # Intersection
    x_left_inter = tf.maximum(x_left_true, x_left_pred)
    x_right_inter = tf.minimum(x_right_true, x_right_pred)
    y_top_inter = tf.maximum(y_top_true, y_top_pred)
    y_bottom_inter = tf.minimum(y_bottom_true, y_bottom_pred)

    inter_width = tf.maximum(0.0, x_right_inter - x_left_inter)
    inter_height = tf.maximum(0.0, y_bottom_inter - y_top_inter)
    inter_area = inter_width * inter_height

    # Union
    union_area = area_true + area_pred - inter_area + 1e-7
    iou = inter_area / union_area

    # Centers and distances
    x_center_true = (x_left_true + x_right_true) / 2.0
    y_center_true = (y_top_true + y_bottom_true) / 2.0
    x_center_pred = (x_left_pred + x_right_pred) / 2.0
    y_center_pred = (y_top_pred + y_bottom_pred) / 2.0

    center_distance = (x_center_pred - x_center_true) ** 2 + (y_center_pred - y_center_true) ** 2

    # Enclosing box
    x_left_enclose = tf.minimum(x_left_true, x_left_pred)
    x_right_enclose = tf.maximum(x_right_true, x_right_pred)
    y_top_enclose = tf.minimum(y_top_true, y_top_pred)
    y_bottom_enclose = tf.maximum(y_bottom_true, y_bottom_pred)

    enclose_diagonal = (x_right_enclose - x_left_enclose) ** 2 + (y_bottom_enclose - y_top_enclose) ** 2 + 1e-7

    # Aspect ratio penalty
    w_true = tf.maximum(1e-7, x_right_true - x_left_true)
    h_true = tf.maximum(1e-7, y_bottom_true - y_top_true)
    w_pred = tf.maximum(1e-7, x_right_pred - x_left_pred)
    h_pred = tf.maximum(1e-7, y_bottom_pred - y_top_pred)

    v = (4 / (3.14159265**2)) * tf.square(tf.atan(w_true / h_true) - tf.atan(w_pred / h_pred))
    alpha = v / (1 - iou + v + 1e-7)

    # CIoU calculation
    ciou = iou - (center_distance / enclose_diagonal) - alpha * v

    # Apply valid mask and penalize invalid predictions
    ciou = ciou * valid_mask - invalid_penalty

    # CIoU loss
    ciou_loss_value = 1 - ciou

    # Regularization term (MSE for bounding box coordinates)
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Combined loss
    loss = tf.reduce_mean(ciou_loss_value) + lambda_reg * mse_loss

    return loss
