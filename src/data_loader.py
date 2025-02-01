import tensorflow as tf

from src.image_utils import data_generator, bbox_data_generator, is_valid_bbox

def create_bbox_dataset(data_dir, max_boxes=10) -> tf.data.Dataset:
    """Creates a TensorFlow dataset with images and their bounding box labels

    Returns:
        tf.data.Dataset: Dataset with images and their bounding box labels
        - Images: (512, 512, 16)
        - Labels: (max_boxes, 4)
    """
    output_sig = (
        tf.TensorSpec(shape=(512, 512, 16), dtype=tf.float32),  # Images
        tf.TensorSpec(shape=(max_boxes, 4), dtype=tf.float32)   # bounding box labels
    )

    return tf.data.Dataset.from_generator(
        lambda: bbox_data_generator(data_dir, max_boxes),
        output_signature=output_sig
    )

def create_dataset() -> tf.data.Dataset: # TODO: Modify to accept other base data dirs
    """
    Creates a TensorFlow dataset with images and their labels with the given dimensions:
        - Images: (512, 512, 16)
        - Labels: (512, 512, 1)
    Each image channel corressponds to a specific hyperspectral frequency image.
    """
    output_sig = (
        tf.TensorSpec(shape=(512, 512, 16), dtype=tf.float32),  # Images
        tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32)   # Labels
    )

    dir = './data/raw_data/STARCOP_train_easy' # Use train easy dataset by default

    return tf.data.Dataset.from_generator(
        lambda: data_generator(dir),
        output_signature=output_sig
    )

def augment_image(image, bboxes, transformation):
    """
    Applies the specified transformation to the image and updates the bounding box accordingly.
    """
    if transformation == "none":
        return image, bboxes

    augmented_bboxes = []

    valid_mask = tf.cast(tf.map_fn(is_valid_bbox, bboxes, dtype=tf.bool), tf.bool)
    valid_mask = tf.expand_dims(valid_mask, axis=-1)  # Shape becomes (10, 1)
    valid_mask = tf.broadcast_to(valid_mask, tf.shape(bboxes))  # Shape (10, 4)

    image_shape = tf.cast(tf.shape(image), tf.float32)

    if transformation == "horizontal_flip":
        image = tf.image.flip_left_right(image)
        augmented_bboxes = tf.where(
            valid_mask,  
            tf.stack([image_shape[1] - bboxes[:, 1] - 1, image_shape[1] - bboxes[:, 0] - 1, bboxes[:, 2], bboxes[:, 3]], axis=1),
            tf.fill(tf.shape(bboxes), -1.0),  # Fill with -1 for invalid boxes
        )
    elif transformation == "vertical_flip":
        image = tf.image.flip_up_down(image)
        augmented_bboxes = tf.where(
            valid_mask,  
            tf.stack([bboxes[:, 0], bboxes[:, 1], image_shape[0] - bboxes[:, 3] - 1, image_shape[0] - bboxes[:, 2] - 1], axis=1),
            tf.fill(tf.shape(bboxes), -1.0),  # Fill with -1 for invalid boxes
        )
    elif transformation == "rotate":
        image = tf.image.rot90(image)  # rotate 90 degrees
        augmented_bboxes = tf.where(
            valid_mask,  
            tf.stack([bboxes[:, 2], bboxes[:, 3], image_shape[1] - bboxes[:, 1] - 1, image_shape[1] - bboxes[:, 0] - 1], axis=1),
            tf.fill(tf.shape(bboxes), -1.0),  # Fill with -1 for invalid boxes
        )
    return image, augmented_bboxes

def augment_dataset(image, bbox, augmentations=["none", "horizontal_flip", "vertical_flip", "rotate"]):
    """
    Applies augmentations to the dataset and combines augmented dataset with the original dataset.
    """
    datasets = []

    for augmentation in augmentations:
        img, box = augment_image(image, bbox, augmentation)
        datasets.append(tf.data.Dataset.from_tensors((img, box)))
    
    return tf.data.Dataset.from_tensor_slices(datasets).flat_map(lambda x: x)

if __name__ == "__main__":
    # testing the shapes of the images and bboxes
    dataset = create_bbox_dataset(data_dir='./data/raw_data/STARCOP_train_easy')

    for image, bbox in dataset.take(3):
        print(f"original bounding box: {bbox}")
        print(f"Original Image Shape: {image.shape}, Original Bbox Shape: {bbox.shape}")

    augmented_dataset = dataset.flat_map(augment_dataset)

    for image, bbox in augmented_dataset.take(3):
        print(f"augmented bounding box: {bbox}")
        print(f"Augmented Image Shape: {image.shape}, Augmented Bbox Shape: {bbox.shape}")

