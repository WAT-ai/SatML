import yaml
from enum import Enum


class PipelineType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class PipelineManager:

    def __init__(self, type, config_path):
        if not isinstance(type, PipelineType):
            raise ValueError(f"Invalid pipeline type: {type}")
        self.type = type
        self.config = self.load_config(config_path)
        self.steps = []

    def load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def build_pipeline(self):
        # Dynamically create pipeline steps based on its type
        if self.type == PipelineType.TRAINING:
            pass
        else:
            pass

    def run(self):
        for step in self.steps:
            step.execute()


