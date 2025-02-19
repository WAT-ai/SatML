import yaml

from config.constants import PipelineType
from src.data_loader.DataLoader import DataLoader


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
            step_configs = self.config.get("steps", [])
            
            for step_config in step_configs:
                class_name = step_config["class"]
                params = step_config["params"]
                
                # Dynamically get class reference and instantiate it
                cls = globals().get(class_name)  
                if cls is None:
                    raise ValueError(f"Class {class_name} not found")
                
                instance = cls(**params)
                self.steps.append(instance)

        else:
            pass

    def run(self):
        for step in self.steps:
            step.execute()


