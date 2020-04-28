import argparse
import numpy as np
from PIL import Image

import torchvision, torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.structures import ImageList
import cv2


def _get_model(args):
    
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model

    model = build_model(cfg) # returns a torch.nn.Module
    weights = args.weights if args.weights!=None else cfg.MODEL.WEIGHTS
    
    DetectionCheckpointer(model).load(weights) # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
    model.train(False) # inference mode
    
    return model

def _get_d2_inputs(image):
    
    img = cv2.imread(args.image)
    img = np.transpose(img,(2,0,1))
    img_tensor = torch.from_numpy(img)
    inputs = [{"image":img_tensor}, {"image":img_tensor}] # inputs is ready
    
    return inputs
    

def run_script(args):
    
    print("start scripting")
    model = _get_model(args)
    
    # Trace parts of composite RCNN model
    # 1. BACKBONE
    scripted = torch.jit.script(model.backbone)
    
    # 2. PROPOSAL NETWORK
    #TODO: this is not implemented currently.
    
    return scripted


def run_trace(args):
    
    print("start tracing")
    model = _get_model(args)
    inputs = _get_d2_inputs(args.image)
    
    print(inputs)

    # Trace parts of composite RCNN model
    # 1. BACKBONE
    images_prep = model.preprocess_image(inputs)
    
    output = model.backbone(images_prep.tensor)
    traced_bcbn = torch.jit.trace(model.backbone, images_prep.tensor)
    
    # 2. PROPOSAL NETWORK
    #TODO: this is not implemented currently.
    
    

if __name__ == "__main__":
    
    # Sagemaker configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    parser.add_argument('--task', default='script', type=str)
    parser.add_argument('--image', default=None, type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--weights', default=None)
    args = parser.parse_args()
    
    if args.task=='script':
        run_script(args)    
    elif args.task=='trace':
        run_trace(args)
        
    

         