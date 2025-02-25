from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    def __init__(self, dataset_dir, batch_size=32, exclude_dirs=[]):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.exclude_dirs = exclude_dirs
        self.dataset = None

    @abstractmethod
    def create_dataset(self):
        pass

    def get_dataset(self):
        return self.dataset

