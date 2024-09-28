import os
import logging

from pathlib import Path


def check_file_exist(file_path: str | os.PathLike) -> bool:
    """Checks if a file with given path exists

    Args:
        file_path (str | os.PathLike): path of file to check

    Returns:
        bool: True if file exists, False otherwise
    """
    if not Path(file_path).exists():
        logging.error(f"File {file_path} does not exist")
        return False
    elif not Path(file_path).is_file():
        logging.error(f"Path {file_path} is not a file")
        return False

    return True


def check_dir_exist(dir_path: str | os.PathLike) -> bool:
    """Checks if a directory with given path exists

    Args:
        dir_path (str | os.PathLike): path of file to check

    Returns:
        bool: True if directory exists, False otherwise
    """
    if not Path(dir_path).exists():
        logging.error(f"dir {dir_path} does not exist")
        return False
    elif not Path(dir_path).is_dir():
        logging.error(f"Path {dir_path} is not a directory")
        return False

    return True


def get_available_datasets(raw_data_dir: str | os.PathLike) -> list[str]:
    """Checks the dataset splits that are available in the given raw data dir

    if all the data is included, the returned list would include the following:
    'test', 'train_easy', 'train_remaining_part1', 'train_remaining_part2', 'train_remaining_part3', 'train_remaining_part4'

    Args:
        raw_data_dir (str | os.PathLike): path to the raw_data directory
        (if you're using the default setup, this should be data/raw_data)

    Returns:
        list: list of strings indicating dataset split names
    """
    available_datasets = []

    if not check_dir_exist(raw_data_dir):
        return available_datasets

    for sub_dir in Path(raw_data_dir).glob('*'):
        if sub_dir.is_dir() and sub_dir.stem.startswith('STARCOP_'):
            dataset_name = sub_dir.stem[len('STARCOP_'):]
            logging.debug(f"Discovered dataset {dataset_name}")
            available_datasets.append(dataset_name)

    return available_datasets
