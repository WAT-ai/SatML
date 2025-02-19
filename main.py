from pipeline.PipelineManager import PipelineManager
from config.constants import PipelineType

if __name__ == "__main__":
    config_path = "./config/bbox_pipeline_config.yaml"

    print("Initializing pipeline manager...")
    pipeline = PipelineManager(PipelineType.TRAINING, config_path)

    print("Creating pipeline flow...")
    pipeline.build_pipeline()

    print("Running pipeline flow...")
    pipeline.run()
