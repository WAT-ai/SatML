import tensorflow as tf
import unittest
from src import data_loader


class TestCreateDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./data/raw_data/STARCOP_train_easy"

    def test_create_dataset(self):
        # Create the dataset
        dataset = data_loader.create_dataset(self.test_dir)

        # Check if dataset is a tf.data.Dataset instance
        self.assertIsInstance(dataset, tf.data.Dataset, "Expected a tf.data.Dataset instance")

        # Check the shape and data type of a single batch
        for sample in dataset.take(1):
            self.assertTrue("image" in sample, "Sample should contain 'image' key")
            self.assertTrue("segmentation_mask" in sample, "Sample should contain 'segmentation_key' key")
            self.assertTrue("directory_path" in sample, "Sample should contain 'directory_path' key")

            self.assertEqual(sample["image"].shape, (512, 512, 16), "Images shape does not match expected dimensions")
            self.assertEqual(
                sample["segmentation_mask"].shape, (512, 512, 1), "Labels shape does not match expected dimensions"
            )

            self.assertEqual(sample["image"].dtype, tf.float32, "Image data should be of type tf.float32")
            self.assertEqual(sample["segmentation_mask"].dtype, tf.float32, "Label data should be of type tf.float32")
            self.assertEqual(
                sample["directory_path"].dtype, tf.string, "directory_path path should be of type tf.string"
            )

            return  # Exit after checking the first sample

        self.fail("No samples were generated from the dataset")

    def test_data_generator(self):
        data_generator = data_loader.data_generator(self.test_dir)

        for sample in data_generator:
            self.assertIsInstance(sample, tuple, "Sample should be a tuple")
            self.assertEqual(len(sample), 3, "Sample should contain 3 elements: image, label, directory")
            images, labels, directory = sample
            self.assertEqual(images.shape, (512, 512, 16), "Image shape does not match expected dimensions")
            self.assertEqual(labels.shape, (512, 512, 1), "Label shape does not match expected dimensions")
            self.assertIsInstance(directory, str, "Directory should be a string")
            self.assertEqual(images.dtype, tf.float32, "Image data should be of type tf.float32")
            self.assertEqual(labels.dtype, tf.float32, "Label data should be of type tf.float32")

            return

        self.fail("No samples were generated from the data generator")

    def test_bbox_data_generator(self):
        max_boxes = 10
        bbox_data_generator = data_loader.bbox_data_generator(self.test_dir, max_boxes)

        for sample in bbox_data_generator:
            self.assertIsInstance(sample, tuple, "Sample should be a tuple")
            self.assertEqual(len(sample), 3, "Sample should contain 3 elements: image, bboxes, directory")
            images, bboxes, directory = sample
            self.assertEqual(images.shape, (512, 512, 16), "Image shape does not match expected dimensions")
            self.assertEqual(bboxes.shape, (max_boxes, 4), "BBoxes shape does not match expected dimensions")
            self.assertIsInstance(directory, str, "Directory should be a string")
            self.assertEqual(images.dtype, tf.float32, "Image data should be of type tf.float32")
            self.assertEqual(bboxes.dtype, tf.int64, "BBoxes data should be of type tf.float32")

            return

        self.fail("No samples were generated from the bbox data generator")



if __name__ == "__main__":
    unittest.main()
