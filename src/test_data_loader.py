import tensorflow as tf
import unittest
from data_loader import create_dataset 

class TestCreateDataset(unittest.TestCase):
    def test_dataset_creation(self):
        # Create the dataset
        dataset = create_dataset()
        
        # Check if dataset is a tf.data.Dataset instance
        self.assertIsInstance(dataset, tf.data.Dataset, "Expected a tf.data.Dataset instance")
        
        # Check the shape and data type of a single batch
        for images, labels in dataset.take(1):
            self.assertEqual(images.shape, (16, 512, 512), "Images shape does not match expected dimensions")
            self.assertEqual(labels.shape, (1, 512, 512), "Labels shape does not match expected dimensions")
            self.assertEqual(images.dtype, tf.float32, "Images dtype should be tf.float32")
            self.assertEqual(labels.dtype, tf.float32, "Labels dtype should be tf.float32")

if __name__ == "__main__":
    unittest.main()
