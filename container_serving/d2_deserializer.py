import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import json
import torch
import pycocotools.mask as mask_util
import numpy as np
from detectron2.structures import Instances, Boxes


def json_to_d2(predictions, device):
    
    pred_dict = json.loads(predictions)
    
    for k, v in pred_dict.items():
        if k=="pred_boxes":
            boxes_to_tensor = torch.FloatTensor(v).to(device)
            pred_dict[k] = Boxes(boxes_to_tensor)
        if k=="scores":
            pred_dict[k] = torch.Tensor(v).to(device)
        if k=="pred_classes":
            pred_dict[k] = torch.Tensor(v).to(device).to(torch.uint8)
        if k=="pred_masks_rle":
            # Convert masks from pycoco RLE format to Detectron2 format
            pred_masks = np.stack([mask_util.decode(rle) for rle in v])
    
    pred_dict["pred_masks"] = torch.Tensor(pred_masks).to(device).to(torch.bool)
    del pred_dict["pred_masks_rle"]
    
    height, width = pred_dict['image_size']
    del pred_dict['image_size']

    
    inst = Instances((height, width,), **pred_dict)
    
    return {'instances':inst}


def d2_to_json(predictions):
    
    instances = predictions["instances"]
    output = {}

    # Iterate over fields in Instances
    for k,v in instances.get_fields().items():
        
        if k in ["scores", "pred_classes"]:
            output[k] = v.tolist()
            
        if k=="pred_boxes":
            output[k] = v.tensor.tolist()
            
        if k=="pred_masks":
            # Convert masks to pycoco binary RLE format to reduce size
            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
            
            v = v.cpu()
            output["pred_masks_rle"] = [mask_util.encode(np.asfortranarray(mask)) for mask in v]
            
            for rle in output["pred_masks_rle"]:
                rle['counts'] = rle['counts'].decode('utf-8')
    
    # Store image size
    output['image_size'] = instances.image_size

    output = json.dumps(output)
    
    return output

    
    
if __name__ == "__main__":
    """
    Test method which serializes Detectron2 predictions to JSON and back.
    """
    IMAGE = cv2.imread("5382403037_73709768a2_z.jpg")
    CFG = get_cfg()
    CFG.merge_from_file("../R101-FPN/mask_rcnn_R_101_FPN_3x.yaml")
    CFG.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    CFG.MODEL.WEIGHTS = "../R101-FPN/R-101.pkl"
    PREDICTOR = DefaultPredictor(CFG)
    PREDICTIONS = PREDICTOR(IMAGE)
    DEVICE='cuda:0'
    
    pred = d2_to_json(PREDICTIONS)
    inst = json_to_d2(pred, DEVICE)
    
#     print(PREDICTIONS["instances"].get_fields()["pred_masks"])
#     print("+++++++++++++++++++++++")
#     print(inst["instances"].get_fields()["pred_masks"])
    
#    # assert that conversion is correct
    assert torch.equal(PREDICTIONS["instances"].get_fields()["pred_masks"], inst["instances"].get_fields()["pred_masks"])
    print(PREDICTIONS["instances"].get_fields()["pred_masks"].shape)
    print(inst["instances"].get_fields()["pred_masks"].shape)


    
    
    