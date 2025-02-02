import tensorflow as tf

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
    """Computes IoU loss for bounding boxes in (x-left, x-right, y-top, y-bottom) format."""
    # Reshape from (batch, n, 4) to (batch * n, 4)
    y_true = tf.reshape(y_true, [-1, 4])
    y_pred = tf.reshape(y_pred, [-1, 4])

    # Check if the ground truth and predicted boxes are all negative (invalid bounding box)
    all_negative_true = tf.reduce_all(y_true < 0, axis=1)
    all_negative_pred = tf.reduce_all(y_pred < 0, axis=1)
    
    # Mask where both boxes are negative
    valid_boxes_mask = tf.logical_and(all_negative_true, all_negative_pred)
    
    # Extract min/max coordinates for valid boxes
    x_min_pred, x_max_pred = y_pred[:, 0], y_pred[:, 1]
    y_min_pred, y_max_pred = y_pred[:, 2], y_pred[:, 3]

    x_min_true, x_max_true = y_true[:, 0], y_true[:, 1]
    y_min_true, y_max_true = y_true[:, 2], y_true[:, 3]

    # Compute intersection for valid boxes
    x_min_inter = tf.maximum(x_min_pred, x_min_true)
    y_min_inter = tf.maximum(y_min_pred, y_min_true)
    x_max_inter = tf.minimum(x_max_pred, x_max_true)
    y_max_inter = tf.minimum(y_max_pred, y_max_true)

    inter_width = tf.maximum(0.0, x_max_inter - x_min_inter)
    inter_height = tf.maximum(0.0, y_max_inter - y_min_inter)
    intersection_area = inter_width * inter_height

    # Compute union for valid boxes
    pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    true_area = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    union_area = pred_area + true_area - intersection_area

    # Compute IoU
    iou = intersection_area / (union_area + 1e-7)  # Small epsilon to prevent division by zero

    # Compute IoU loss
    iou_loss = 1 - iou

    # Set IoU loss to 0 for invalid boxes, keep normal loss for valid boxes
    final_loss = tf.where(valid_boxes_mask, 0.0, iou_loss)

    return tf.reduce_mean(final_loss)  # Reduce over batch
