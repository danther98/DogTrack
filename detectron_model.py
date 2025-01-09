"""
Script to detect and segment dogs using a local `.pkl` model file and a local YAML config file,
if available. If a local config does NOT exist, optionally fall back to the Detectron2 model zoo.
"""

import os
import cv2
import torch
import numpy as np
from typing import Tuple, Optional
from detect.detectron2.detectron2.config import get_cfg
from detect.detectron2.detectron2 import model_zoo
from detect.detectron2.detectron2.engine import DefaultPredictor
from detect.detectron2.detectron2.data import MetadataCatalog


def draw_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlays a single segmentation mask on the frame with a specified color and transparency.
    """
    bool_mask = mask.astype(bool)
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[bool_mask] = color
    return cv2.addWeighted(frame, 1.0, overlay, alpha, 0)


def setup_detectron2_model(
    local_config_path: str,
    local_weights_path: str,
    fallback_config_zoo: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    score_thresh: float = 0.7
) -> DefaultPredictor:
    """
    Sets up a Detectron2 Mask R-CNN model using a local .yaml config if present.
    If the local config file doesn't exist, optionally fall back to a model zoo config.

    Args:
        local_config_path (str): Path to your local .yaml config file (if it exists).
        local_weights_path (str): Path to your downloaded .pkl weights file.
        fallback_config_zoo (str): Model zoo config path if local YAML is missing.
        score_thresh (float, optional): Threshold for predictions. Defaults to 0.7.

    Returns:
        DefaultPredictor: A Detectron2 predictor ready for inference.
    """
    cfg = get_cfg()

    if os.path.isfile(local_config_path):
        print(f"[INFO] Using local config file: {local_config_path}")
        cfg.merge_from_file(local_config_path)
    else:
        print(f"[WARNING] Local config '{local_config_path}' not found.")
        print(f"[INFO] Falling back to model zoo config: {fallback_config_zoo}")
        cfg.merge_from_file(model_zoo.get_config_file(fallback_config_zoo))

    # Point to your local .pkl weights
    cfg.MODEL.WEIGHTS = local_weights_path

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return DefaultPredictor(cfg)


def track_dogs_in_webcam(
    local_config_path: str,
    local_weights_path: str,
    conf_thresh: float = 0.7,
    fallback_config_zoo: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    webcam_id: int = 0
) -> None:
    """
    Streams frames from a webcam and applies a Detectron2 Mask R-CNN model
    (with local .yaml config if available, and local .pkl weights) to detect
    and segment dogs in real-time.

    Args:
        local_config_path (str): Path to the local config .yaml file.
        local_weights_path (str): Path to the .pkl weights you downloaded.
        conf_thresh (float, optional): Confidence threshold. Defaults to 0.7.
        fallback_config_zoo (str): Model zoo config to use if local YAML is missing.
        webcam_id (int, optional): Webcam device ID. Defaults to 0.

    Returns:
        None
    """
    predictor = setup_detectron2_model(
        local_config_path=local_config_path,
        local_weights_path=local_weights_path,
        fallback_config_zoo=fallback_config_zoo,
        score_thresh=conf_thresh
    )

    # The dataset name is typically from the config's 'DATASETS.TRAIN'
    metadata = MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0])
    class_names = metadata.thing_classes  # e.g., ["person", "bicycle", ... , "dog", ...]

    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {webcam_id}.")
        return

    # Identify index of 'dog' if present
    dog_class_index: Optional[int] = None
    if "dog" in class_names:
        dog_class_index = class_names.index("dog")

    print(f"Running on device: {predictor.cfg.MODEL.DEVICE}")
    print(f"Using weights from: {local_weights_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break

            outputs = predictor(frame)
            instances = outputs["instances"].to("cpu")

            # Extract predictions
            pred_classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
            pred_boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            pred_masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
            scores = instances.scores.numpy() if instances.has("scores") else []

            # Draw only the 'dog' predictions
            for i, cls_id in enumerate(pred_classes):
                if dog_class_index is not None and cls_id == dog_class_index and scores[i] >= conf_thresh:
                    (x1, y1, x2, y2) = pred_boxes[i].astype(int)
                    mask = pred_masks[i]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Dog: {scores[i]:.2f}",
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    # Overlay mask
                    frame = draw_mask(frame, mask, color=(0, 255, 0), alpha=0.4)

            cv2.imshow("Dog Detection (Local YAML + Local Weights)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage:
    # 1. Provide a local config .yaml file (if it exists) and a local .pkl weights path.
    # 2. If the .yaml file doesn't exist, it will fall back to the model zoo config.

    LOCAL_CONFIG_PATH = r"models/mask_rcnn_R_101_FPN_1x/mask_rcnn_R_101_FPN_1x.yaml"  # local config
    LOCAL_WEIGHTS_PATH = r"models/mask_rcnn_R_101_FPN_1x/model_final_824ab5.pkl"
    FALLBACK_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    track_dogs_in_webcam(
        local_config_path=LOCAL_CONFIG_PATH,
        local_weights_path=LOCAL_WEIGHTS_PATH,
        conf_thresh=0.7,
        fallback_config_zoo=FALLBACK_CONFIG,
        webcam_id=0
    )
