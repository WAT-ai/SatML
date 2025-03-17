from enum import Enum

IMAGE_FILE_NAMES = (
        "TOA_AVIRIS_460nm.tif",
        "TOA_AVIRIS_550nm.tif",
        "TOA_AVIRIS_640nm.tif",
        "TOA_AVIRIS_2004nm.tif",
        "TOA_AVIRIS_2109nm.tif",
        "TOA_AVIRIS_2310nm.tif",
        "TOA_AVIRIS_2350nm.tif",
        "TOA_AVIRIS_2360nm.tif",
        "TOA_WV3_SWIR1.tif",
        "TOA_WV3_SWIR2.tif",
        "TOA_WV3_SWIR3.tif",
        "TOA_WV3_SWIR4.tif",
        "TOA_WV3_SWIR5.tif",
        "TOA_WV3_SWIR6.tif",
        "TOA_WV3_SWIR7.tif",
        "TOA_WV3_SWIR8.tif")

class PipelineType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"

class DatasetType(Enum):
    SEGMENTATION = "segmentation"
    BOUNDING_BOX = "bounding_box"