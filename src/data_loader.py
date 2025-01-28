import tensorflow as tf

from src.image_utils import data_generator

def create_dataset() -> tf.data.Dataset: # TODO: Modify to accept other base data dirs
    """
    Creates a TensorFlow dataset with images and their bounding box labels:
        - Images: (512, 512, 16)
        - Labels: (max_boxes, 4)
    Each image channel corressponds to a specific hyperspectral frequency image.
    """
    output_sig = (
        tf.TensorSpec(shape=(512, 512, 16), dtype=tf.float32),  # Images
        tf.TensorSpec(shape=(4), dtype=tf.float32)   # bounding box labels
    )

    dir = './data/raw_data/STARCOP_train_easy' # Use train easy dataset by default

    return tf.data.Dataset.from_generator(
        lambda: data_generator(dir),
        output_signature=output_sig
    )

def augment_image(image, bbox, transformation):
    """
    Applies the specified transformation to the image and updates the bounding box accordingly.
    """
    if transformation == "horizontal_flip":
        image = tf.image.flip_left_right(image)
        bbox = tf.stack([1 - bbox[2], bbox[1], 1 - bbox[0], bbox[3]])  # reverse x
    elif transformation == "vertical_flip":
        image = tf.image.flip_up_down(image)
        bbox = tf.stack([bbox[0], 1 - bbox[3], bbox[2], 1 - bbox[1]])  # reverse y
    elif transformation == "rotate":
        image = tf.image.rot90(image)  # rotate 90 degrees
        bbox = tf.stack([bbox[1], 1 - bbox[2], bbox[3], 1 - bbox[0]])
    # will add rotations of 180 and 270 degrees    
    return image, bbox

def augment_dataset(dataset, augmentations=["horizontal_flip", "vertical_flip", "rotate"]):
    """
    Applies augmentations to the dataset and combines augmented dataset with the original dataset.
    """
    all_imgs = []
    all_bboxes = []

    for image, bbox in dataset:
        all_imgs.append(image)
        all_bboxes.append(bbox)
        for aug in augmentations:
            aug_image, aug_bbox = augment_image(image, bbox, aug)
            all_imgs.append(aug_image)
            all_bboxes.append(aug_bbox)
    
    return tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(all_imgs), tf.convert_to_tensor(all_bboxes)))

if __name__ == "__main__":
    # testing the shapes of the images and bboxes
    dataset = create_dataset()

    for image, bbox in dataset.take(3):
        print(f"original bounding box: {bbox}")
        print(f"Original Image Shape: {image.shape}, Original Bbox Shape: {bbox.shape}")

    augmented_dataset = augment_dataset(dataset.take(3))

    for image, bbox in augmented_dataset.take(12):
        print(f"augmented bounding box: {bbox}")
        print(f"Augmented Image Shape: {image.shape}, Augmented Bbox Shape: {bbox.shape}")
