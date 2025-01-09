import cv2
import torch
import numpy as np
from PIL import Image
from typing import Any, Tuple
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def draw_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> np.ndarray:
    bool_mask = mask > 0.5
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[bool_mask] = color
    return cv2.addWeighted(frame, 1.0, overlay, alpha, 0)

def track_dogs_in_webcam(conf_thresh: float = 0.5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Use the newer weights API
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1).to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    transform = transforms.ToTensor()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(Image.fromarray(rgb_frame)).to(device)

        with torch.no_grad():
            predictions = model([input_tensor])

        pred = predictions[0]

        boxes = pred['boxes']
        labels = pred['labels']
        scores = pred['scores']
        masks = pred['masks']

        for i, label_idx in enumerate(labels):
            label_value = label_idx.item()

            # Check index range to avoid IndexError
            if 0 <= label_value < len(COCO_INSTANCE_CATEGORY_NAMES):
                label_name = COCO_INSTANCE_CATEGORY_NAMES[label_value]
            else:
                # Skip or handle out-of-range index
                continue

            if scores[i] > conf_thresh and label_name == 'dog':
                (x1, y1, x2, y2) = boxes[i].int().tolist()
                mask = masks[i, 0].cpu().numpy()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Dog: {scores[i]:.2f}", (x1, max(y1-5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                frame = draw_mask(frame, mask, color=(0, 255, 0), alpha=0.4)

        cv2.imshow('Dog Detection (Mask R-CNN)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_dogs_in_webcam(conf_thresh=0.7)
