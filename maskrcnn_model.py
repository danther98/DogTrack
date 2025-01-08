#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=======================
Visualization utilities
=======================

.. note::
    Try on `Colab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_visualization_utils.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_others_plot_visualization_utils.py>` to download the full example code.

This example illustrates some of the utilities that torchvision offers for
visualizing images, bounding boxes, segmentation masks and keypoints.
"""

# sphinx_gallery_thumbnail_path = "../../gallery/assets/visualization_utils_thumbnail2.png"

import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import (
    make_grid,
    draw_bounding_boxes,
    draw_segmentation_masks,
    draw_keypoints,
)
from torchvision.io import decode_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from pathlib import Path
from torchvision.transforms.functional import resize, rotate

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    """
    Utility to display a list of images (or a single image) using matplotlib.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach() if isinstance(img, torch.Tensor) else img
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def main():
    # Visualizing a grid of images
    # ----------------------------
    # The :func:`~torchvision.utils.make_grid` function can be used to create a
    # tensor that represents multiple images in a grid. This util requires a single
    # image of dtype ``uint8`` as input.

    # Load and decode images
    # Load and decode images
    dog1_int = decode_image(str("IMG_4088.JPG"))
    dog2_int = decode_image(str("IMG_4090.JPG"))

    # Resize images to the same size
    target_size = (1280, 720)  # Example target size (height, width)
    dog1_int = resize(dog1_int, target_size)
    dog2_int = resize(dog2_int, target_size)

    # Rotate dog1_int by 90 degrees
    dog1_int = rotate(dog1_int, 270)

    # Convert images to uint8
    dog1_int = dog1_int.to(torch.uint8)
    dog2_int = dog2_int.to(torch.uint8)

    # Create a list of images
    dog_list = [dog1_int, dog2_int]

    grid = make_grid(dog_list)
    show(grid)

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    images = [transforms(d) for d in dog_list]

    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()

    output = model(images)
    print(output)

    dog1_output = output[0]
    dog1_masks = dog1_output["masks"]
    print(
        f"shape = {dog1_masks.shape}, dtype = {dog1_masks.dtype}, "
        f"min = {dog1_masks.min()}, max = {dog1_masks.max()}"
    )

    print("For the first dog, the following instances were detected:")
    print([weights.meta["categories"][label] for label in dog1_output["labels"]])

    # Convert the probabilities into boolean values via threshold
    proba_threshold = 0.5
    dog1_bool_masks = dog1_output["masks"] > proba_threshold
    dog1_bool_masks = dog1_bool_masks.squeeze(1)
    # show(draw_segmentation_masks(dog1_int, dog1_bool_masks, alpha=0.9))

    print(dog1_output["scores"])

    # Filter by score
    score_threshold = 0.75
    boolean_masks = [
        out["masks"][out["scores"] > score_threshold] > proba_threshold
        for out in output
    ]

    dogs_with_masks = [
        draw_segmentation_masks(img, mask.squeeze(1))
        for img, mask in zip(dog_list, boolean_masks)
    ]
    show(dogs_with_masks)


if __name__ == "__main__":
    main()
