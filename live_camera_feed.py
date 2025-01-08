import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms.functional import resize

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach() if isinstance(img, torch.Tensor) else img
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def process_frame(frame, model, transforms, proba_threshold=0.3, score_threshold=0.5):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    frame_tensor = resize(frame_tensor, (720, 1280))
    frame_tensor = frame_tensor.to(torch.uint8)

    transformed_frame = transforms(frame_tensor)
    output = model([transformed_frame])[0]

    print("Model output:", output)  # Debug statement

    masks = output["masks"]
    labels = output["labels"]
    scores = output["scores"]

    person_masks = masks[(labels == 1) & (scores > score_threshold)]
    bool_masks = person_masks > proba_threshold
    bool_masks = bool_masks.squeeze(1)

    result_frame = draw_segmentation_masks(frame_tensor, bool_masks, alpha=0.9)

    return result_frame


def main():
    cap = cv2.VideoCapture(0)
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Original Frame", frame)  # Visualize original frame

        result_frame = process_frame(frame, model, transforms)
        result_frame = result_frame.permute(1, 2, 0).numpy()
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Live Feed", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
