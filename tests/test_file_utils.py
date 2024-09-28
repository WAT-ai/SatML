import unittest
import pandas as pd

from unittest.mock import patch, MagicMock
from pathlib import Path

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

    def test_get_dataset_path(self):
        raw_data_dir = "data/raw_data"
        dataset_name = "test"
        expected_path = Path(raw_data_dir).joinpath(f'STARCOP_{dataset_name}').as_posix()

        result = file_utils.get_dataset_path(raw_data_dir, dataset_name)
        self.assertEqual(result, expected_path)

    @patch('src.file_utils.check_file_exist', return_value=True)
    @patch('pandas.read_csv')
    def test_read_dataset_csv_with_csv_file_path(self, mock_read_csv, mock_check_file_exist):
        # Simulate valid csv_file_path scenario
        csv_file_path = "data/raw_data/test.csv"
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_read_csv.return_value = mock_df

        result = file_utils.read_dataset_csv(csv_file_path=csv_file_path)

        mock_check_file_exist.assert_called_once_with(csv_file_path)
        mock_read_csv.assert_called_once_with(csv_file_path)
        self.assertEqual(result, mock_df)

    @patch('src.file_utils.get_dataset_path')
    @patch('src.file_utils.check_file_exist', return_value=True)
    @patch('pandas.read_csv')
    def test_read_dataset_csv_with_raw_data_dir_and_dataset_name(self, mock_read_csv, mock_check_file_exist, mock_get_dataset_path):
        # Simulate valid raw_data_dir and dataset_name scenario
        raw_data_dir = "data/raw_data"
        dataset_name = "test"
        dataset_path = "data/raw_data/STARCOP_test"
        mock_get_dataset_path.return_value = dataset_path
        csv_file_path = Path(dataset_path).joinpath(f'{dataset_name}.csv')
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_read_csv.return_value = mock_df

        result = file_utils.read_dataset_csv(raw_data_dir=raw_data_dir, dataset_name=dataset_name)

        mock_get_dataset_path.assert_called_once_with(raw_data_dir, dataset_name)
        mock_check_file_exist.assert_called_once_with(csv_file_path)
        mock_read_csv.assert_called_once_with(csv_file_path)
        self.assertEqual(result, mock_df)

    def test_read_dataset_csv_raises_value_error(self):
        # Test when both csv_file_path and raw_data_dir/dataset_name are None
        with self.assertRaises(ValueError):
            file_utils.read_dataset_csv()

    @patch('src.file_utils.check_file_exist', return_value=False)
    def test_read_dataset_csv_raises_file_not_found(self, mock_check_file_exist):
        # Simulate file not found scenario
        raw_data_dir = "data/raw_data"
        dataset_name = "nonexistent"
        dataset_path = Path(raw_data_dir).joinpath(f'STARCOP_{dataset_name}')
        csv_file_path = Path(dataset_path).joinpath(f'{dataset_name}.csv')

        with self.assertRaises(FileNotFoundError):
            file_utils.read_dataset_csv(raw_data_dir=raw_data_dir, dataset_name=dataset_name)

        mock_check_file_exist.assert_called_once_with(csv_file_path)