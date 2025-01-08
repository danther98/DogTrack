from maskrcnn_tf2.mrcnn import model
import maskrcnn_tf2
from maskrcnn_config import SimpleConfig
import os

model = maskrcnn_tf2.mrcnn.model.MaskRCNN(
    mode="inference", config=SimpleConfig(), model_dir=os.getcwd()
)


model.keras_model.summary()
