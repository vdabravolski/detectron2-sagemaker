"""Code used for sagemaker batch transform jobs"""
import os
from typing import BinaryIO, Mapping
import json
import logging
import sys

import numpy as np
import cv2
import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode

##############
# Macros
##############

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

##########
# Deploy
##########
def _load_from_bytearray(request_body: BinaryIO) -> np.ndarray:
    npimg = np.frombuffer(request_body, np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)


def model_fn(model_dir: str) -> DefaultPredictor:
    """
    Load trained model

    Args:
        model_dir (str): S3 location of the model directory

    Returns:
        nn.Module: trained PyTorch model
    """
    with open(os.path.join(model_dir, "config.json")) as fid:
        cfg = CfgNode(json.load(fid))

    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return DefaultPredictor(cfg)


def input_fn(request_body: BinaryIO, request_content_type: str) -> np.ndarray:
    """
    Parse input data

    Args:
        request_body (BinaryIO): encoded input image
        request_content_type (str): type of content

    Raises:
        ValueError: ValueError if the content type is not 'application/x-image'

    Returns:
        torch.Tensor: input image Tensor
    """
    if request_content_type == "application/x-image":
        np_image = _load_from_bytearray(request_body)
    else:
        raise ValueError(f"Type [{request_content_type}] not support this type yet")
    return np_image


def predict_fn(input_object: np.ndarray, predictor: DefaultPredictor) -> Mapping:
    """
    Run Detectron2 prediction

    Args:
        input_object (np.ndarray): input image
        predictor (DefaultPredictor): Detectron2 default predictor (see Detectron2 documentation
            for details)

    Returns:
        [Mapping]: a dictionary that contains absolute
    """
    LOGGER.info(f"Prediction on image of shape {input_object.shape}")
    outputs = predictor(input_object)
    fmt_out = {
        "image_height": input_object.shape[0],
        "image_width": input_object.shape[1],
        "pred_boxes": outputs["instances"].pred_boxes.tensor.tolist(),
        "scores": outputs["instances"].scores.tolist(),
        "pred_classes": outputs["instances"].pred_classes.tolist(),
    }
    LOGGER.info(f"Number of detected boxes: {len(fmt_out['pred_boxes'])}")
    return fmt_out


# pylint: disable=unused-argument
def output_fn(predictions, response_content_type):
    """Serialize the prediction result into the desired response content type"""
    return json.dumps(predictions)
