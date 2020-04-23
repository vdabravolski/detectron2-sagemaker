# This is default implementation of inference_handler: 
# https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py
# SM specs: https://sagemaker.readthedocs.io/en/stable/using_pytorch.html


# TODO list
# 1. add support of multi-GPU instances - if GPU devices > 1, do round robin
# 2. do we need to support checkpoints (optimizers, LR etc.)

import os
import io
import argparse
import logging
import sys
import pickle    
from yacs.config import CfgNode as CN
import numpy as np

import torch
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
import detectron2.data.transforms as T
from detectron2.config import get_cfg


from sagemaker_inference import content_types, decoder, default_inference_handler, encoder
from sagemaker.content_types import CONTENT_TYPE_JSON, CONTENT_TYPE_CSV, CONTENT_TYPE_NPY # TODO: for local debug only. Remove or comment when deploying remotely.
from six import StringIO, BytesIO  # TODO: for local debug only. Remove or comment when deploying remotely.

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _get_predictor(config_path, model_path):
    
    cfg = get_cfg()
    cfg.merge_from_file(config_path) # get baseline parameters from YAML config
    cfg.MODEL.WEIGHTS = model_path

    pred = DefaultPredictor(cfg)
    logger.info(cfg)
    eval_results = pred.model.eval()
    #logger.debug(eval_results)
    
    pred.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    checkpointer = DetectionCheckpointer(pred.model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    pred.transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, 
                                               cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

    pred.input_format = cfg.INPUT.FORMAT
    assert pred.input_format in ["RGB", "BGR"], pred.input_format
    return pred



def model_fn(model_dir):
    """
    Deserialize and load D2 model. This method is called automatically by Sagemaker.
    model_dir is location where your trained model will be downloaded.
    """
    
    logger.info("Deserializing Detectron2 model...")
    
    # Restoring trained model, take a first .yaml and .pth/.pkl file in the model directory
    for file in os.listdir(model_dir):
        # looks up for yaml file with model config
        if file.endswith(".yaml"):
            config_path = os.path.join(model_dir, file)
        # looks up for *.pkl or *.pth files with model weights
        if file.endswith(".pth") or file.endswith(".pkl"):
            model_path = os.path.join(model_dir, file)
            
    logger.info(f"Using config file {config_path}")
    logger.info(f"Using model weights from {model_path}")            
    
    pred = _get_predictor(config_path,model_path)
    logger.info(f"Deserialization completed ...")
    
    return pred


def input_fn(input_data, content_type=CONTENT_TYPE_NPY):
    """
    Converts image from NPY format to numpy.
    """
    
    logger.info("Handling inputs...")
    image_np = decoder.decode(input_data, content_type)
    
    return image_np


def predict_fn(inputs, predictor):
    # accroding to D2 rquirements: https://detectron2.readthedocs.io/tutorials/models.html
    logger.info("Doing predictions...")
    outputs = predictor(inputs)
    logger.debug(outputs)
    return outputs

def output_fn(predictions, response_content_type=None):
    logger.info("processing outputs...")
    pickled_outputs = pickle.dumps(predictions)
    stream = io.BytesIO(pickled_outputs)
    return stream.getvalue()


if __name__ == "__main__":
    """
    Test method to replicate sequence of calls at inference endpoint. Keep it for local debugging. 
    This code won't be executed on the remote Sagemaker endpoint.
    """
    
    from PIL import Image
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default="../coco_sample.jpg", help='point to test image from coco2017 dataset')
    parser.add_argument('--model-dir', type=str, default="../", help='directory with model weights and configs')
    args = parser.parse_args()
    
    #1. Get the image
    image = Image.open(args.image)
    image_np = np.asarray(image)
        
    # 2. Serialize the data
    image_npy = encoder.encode(image_np, CONTENT_TYPE_NPY)
    
    ##### simulate sending over the wire ######
    
    # 3. Deserialize the data
    image_np = input_fn(image_npy, CONTENT_TYPE_NPY)
    
    # 4. Deserialize the model
    predictor = model_fn(args.model_dir)

    # 5. Do prediction and return output
    predictions = predict_fn(image_np, predictor)
    
    # 6. Serialize D2 custom output to binary format for response body
    outputs = output_fn(predictions)
    
    
    

