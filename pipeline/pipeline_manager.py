import yaml
import importlib

from config.constants import PipelineType

CLASS_MAPPING = {
    "BoundingBoxDataLoader": "src.data_loader.bounding_box_data_loader.BoundingBoxDataLoader",
    "SegmentationDataLoader": "src.data_loader.segmentation_data_loader.SegmentationDataLoader",
    "BoundingBoxProcessor": "src.processor.bounding_box_processor.BoundingBoxProcessor",
    "BoundingBoxModel": "models.bounding_box_model.BoundingBoxModel",
}

class PipelineManager:

    def __init__(self, type, config_path):
        if not isinstance(type, PipelineType):
            raise ValueError(f"Invalid pipeline type: {type}")
        self.type = type
        self.config = self.load_config(config_path)
        self.data_loader = self._load_component("DataLoader")
        self.processor = self._load_component("Processor")
        self.model = self._load_component("Model")

    def _load_component(self, key):
        """Dynamically loads a class from CLASS_MAPPING based on the YAML config."""
        if key not in self.config:
            raise ValueError(f"Missing '{key}' section in config file.")

        key_type = self.config[key]["type"]
        class_path = CLASS_MAPPING.get(key_type)  # Get class path from mapping
        
        if not class_path:
            raise ValueError(f"Unknown type '{key_type}' for '{key}' component.")

        params = self.config[key].get("params", {})  # Extract parameters
        module_name, class_name = class_path.rsplit(".", 1)  # Split module & class
        module = importlib.import_module(module_name)  # Import module dynamically
        cls = getattr(module, class_name)  # Get class from module

        return cls(**params)  # Instantiate and return instance

    def load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)
