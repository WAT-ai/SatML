import unittest
import os

from src.data_utils import get_easy_ids

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        """Create a sample CSV file to use in the tests."""
        self.test_file = "test_easy.csv"
        with open(self.test_file, "w") as f:
            f.write("id,difficulty\n")
            f.write("1, easy\n")
            f.write("2, hard\n")
            f.write("3, easy\n")

    def tearDown(self):
        """Remove the sample CSV file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_get_easy_ids(self):
        """Test that get_easy_ids returns the correct IDs."""
        result = get_easy_ids(self.test_file)
        expected = ["1", "3"]
        self.assertEqual(result, expected)

    def test_empty_file(self):
        """Test that get_easy_ids works with an empty file."""
        with open(self.test_file, "w") as f:
            f.write("id,difficulty\n")  # Empty data except headers

        result = get_easy_ids(self.test_file)
        expected = []
        self.assertEqual(result, expected)

    def test_invalid_file(self):
        """Test that get_easy_ids raises an error on invalid files."""
        with self.assertRaises(ValueError):
            get_easy_ids("non_existent_file.csv")


if __name__ == "__main__":
    unittest.main()
