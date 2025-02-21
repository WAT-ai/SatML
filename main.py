from pipeline.PipelineManager import PipelineManager
from config.constants import PipelineType

if __name__ == "__main__":
    config_path = "./config/bbox_pipeline_config.yaml"

    print("Initializing pipeline manager...")
    pipeline = PipelineManager(PipelineType.TRAINING, config_path)

    print("Loading dataset...")
    pipeline.data_loader.create_dataset()
    data = pipeline.data_loader.get_dataset()

    print("Processing dataset...")
    pipeline.processor.preprocess(data)

