import unittest
from unittest.mock import patch, MagicMock

from src import file_utils


class TestFileUtils(unittest.TestCase):
    
    @patch('src.file_utils.Path.exists')
    @patch('src.file_utils.Path.is_file')
    def test_check_file_exist_file_exists(self, mock_is_file, mock_exists):
        # Simulate file exists and is a file
        mock_exists.return_value = True
        mock_is_file.return_value = True

        result = file_utils.check_file_exist("test.txt")
        self.assertTrue(result)
        mock_exists.assert_called_once_with()
        mock_is_file.assert_called_once_with()

    @patch('src.file_utils.Path.exists')
    def test_check_file_exist_file_does_not_exist(self, mock_exists):
        # Simulate file does not exist
        mock_exists.return_value = False

        result = file_utils.check_file_exist("nonexistent.txt")
        self.assertFalse(result)
        mock_exists.assert_called_once_with()

    @patch('src.file_utils.Path.exists')
    @patch('src.file_utils.Path.is_file')
    def test_check_file_exist_is_not_file(self, mock_is_file, mock_exists):
        # Simulate path exists but is not a file
        mock_exists.return_value = True
        mock_is_file.return_value = False

        result = file_utils.check_file_exist("directory_instead_of_file")
        self.assertFalse(result)
        mock_exists.assert_called_once_with()
        mock_is_file.assert_called_once_with()

    @patch('src.file_utils.Path.exists')
    @patch('src.file_utils.Path.is_dir')
    def test_check_dir_exist_directory_exists(self, mock_is_dir, mock_exists):
        # Simulate directory exists and is a directory
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        result = file_utils.check_dir_exist("test_directory")
        self.assertTrue(result)
        mock_exists.assert_called_once_with()
        mock_is_dir.assert_called_once_with()

    @patch('src.file_utils.Path.exists')
    def test_check_dir_exist_directory_does_not_exist(self, mock_exists):
        # Simulate directory does not exist
        mock_exists.return_value = False

        result = file_utils.check_dir_exist("nonexistent_directory")
        self.assertFalse(result)
        mock_exists.assert_called_once_with()

    @patch('src.file_utils.Path.exists')
    @patch('src.file_utils.Path.is_dir')
    def test_check_dir_exist_is_not_directory(self, mock_is_dir, mock_exists):
        # Simulate path exists but is not a directory
        mock_exists.return_value = True
        mock_is_dir.return_value = False

        result = file_utils.check_dir_exist("file_instead_of_directory")
        self.assertFalse(result)
        mock_exists.assert_called_once_with()
        mock_is_dir.assert_called_once_with()

    @patch('src.file_utils.check_dir_exist', return_value=True)
    @patch('src.file_utils.Path.glob')
    def test_get_available_datasets(self, mock_glob, mock_check_dir_exist):
        # Simulate directory contents
        mock_subdir1 = MagicMock()
        mock_subdir1.is_dir.return_value = True
        mock_subdir1.stem = 'STARCOP_test'

        mock_subdir2 = MagicMock()
        mock_subdir2.is_dir.return_value = True
        mock_subdir2.stem = 'STARCOP_train_easy'

        mock_glob.return_value = [mock_subdir1, mock_subdir2]

        expected_datasets = ['test', 'train_easy']
        result = file_utils.get_available_datasets("raw_data_dir")

        self.assertEqual(result, expected_datasets)
        mock_glob.assert_called_once_with('*')

    @patch('src.file_utils.check_dir_exist', return_value=False)
    def test_get_available_datasets_invalid_dir(self, mock_check_dir_exist):
        # Test when directory check fails
        result = file_utils.get_available_datasets("invalid_dir")
        self.assertEqual(result, [])  # Should return empty list