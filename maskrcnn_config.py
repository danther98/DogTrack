from maskrcnn_tf2.mrcnn import config


class SimpleConfig(config.Config):
    NAME = "coco_inference"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 81
