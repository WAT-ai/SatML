import tensorflow as tf

from src.image_utils import data_generator, get_single_bounding_box


class SegmentPipeline:
    def __init__(self, channels_of_interest, num_classes, target_shape):
        self.channels_of_interest = channels_of_interest
        self.num_classes = num_classes
        self.num_input_channels = len(channels_of_interest) if channels_of_interest is not None else 16
        self.target_shape = target_shape

    def create_dataset(self, dir: str) -> tf.data.Dataset:
        """
        Creates a TensorFlow dataset with images and their labels with the given dimensions:
            - Images: (512, 512, 16)
            - Labels: (512, 512, 1)
        Each image channel corressponds to a specific hyperspectral frequency image.
        """
        output_sig = (
            tf.TensorSpec(shape=(512, 512, self.num_input_channels), dtype=tf.float32),  # Images
            tf.TensorSpec(shape=(512, 512, self.num_classes), dtype=tf.float32),  # Labels
        )

        file_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(dir, self.channels_of_interest),
            output_signature=output_sig,
        )

        cropped_dataset = self.generate_cropped_dataset(file_dataset, get_single_bounding_box)

        return cropped_dataset

    def generate_cropped_dataset(
        self, original_dataset: tf.data.Dataset, bounding_box_function: callable
    ) -> tf.data.Dataset:
        """
        Generate a new dataset by cropping each image into multiple images based on bounding boxes.

        Args:
            original_dataset (tf.data.Dataset): The original dataset created with tf.data.Dataset.from_generator.
            bounding_box_function (callable): A function that takes an image as input and returns a list of bounding boxes,
                                            each represented as (x_min, y_min, x_max, y_max).

        Returns:
            tf.data.Dataset: A new dataset containing cropped images.
        """

        def crop_images(image, label):
            # Get bounding boxes for the current image
            bounding_boxes = [bounding_box_function(label[..., 0])]

            # Crop the image for each bounding box
            cropped_images = []
            cropped_labels = []
            for box in bounding_boxes:
                x_min, x_max, y_min, y_max = box

                if x_min == x_max or y_min == y_max:
                    continue

                cropped_image = tf.image.crop_to_bounding_box(image, y_min, x_min, y_max - y_min, x_max - x_min)
                cropped_label = tf.image.crop_to_bounding_box(label, y_min, x_min, y_max - y_min, x_max - x_min)

                # pad and resize the cropped image and label
                cropped_image = tf.image.resize_with_pad(cropped_image, self.target_shape[0], self.target_shape[1])
                cropped_label = tf.image.resize_with_pad(cropped_label, self.target_shape[0], self.target_shape[1])

                cropped_images.append(cropped_image)
                cropped_labels.append(cropped_label)  # Adjust the label as needed

            # Convert lists to tensors
            return tf.stack(cropped_images), tf.stack(cropped_labels)

        # Wrap the cropping function in a tf.py_function
        def process_images(image, label):
            images, labels = tf.py_function(crop_images, [image, label], [tf.float32, tf.float32])
            return tf.data.Dataset.from_tensor_slices((images, labels))

        # Apply the cropping function to each element and flatten the dataset
        cropped_dataset = original_dataset.flat_map(process_images)

        return cropped_dataset


if __name__ == "__main__":
    pipeline = SegmentPipeline(channels_of_interest=None, num_classes=1, target_shape=(256, 256))
    dataset = pipeline.create_dataset("./data/raw_data/STARCOP_train_easy")

    for images, labels in dataset.take(1):
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Images dtype: {images.dtype}")
        print(f"Labels dtype: {labels.dtype}")
        break
