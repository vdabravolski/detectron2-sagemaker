import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import json
import torch


from detectron2.structures import Instances, Boxes


def json_to_d2(predictions, device):
    
    pred_dict = json.loads(predictions)
    height, width = pred_dict['image_size']
    del pred_dict['image_size']
    
    for k, v in pred_dict.items():
        if k=="pred_boxes":
            boxes_to_tensor = torch.FloatTensor(pred_dict[k]).to(device)
            pred_dict[k] = Boxes(boxes_to_tensor)
        if k=="scores":
            pred_dict[k] = torch.Tensor(pred_dict[k]).to(device)
        if k=="pred_classes":
            pred_dict[k] = torch.Tensor(pred_dict[k]).to(device).to(torch.uint8)

# TODO: Don't save pred_masks for now. If saved, the response size will violate netty limits.
#         if k=="pred_masks":
#             pred_dict[k] = torch.Tensor(pred_dict[k]).to(device).to(torch.bool)
            
    inst = Instances((height, width,), **pred_dict)
    
    return {'instances':inst}


def d2_to_json(predictions):
    
    instances = predictions["instances"]
    output = {}

    # Iterate over fields in Instances
    for field, value in instances.get_fields().items():
        if field=="scores":
            output[field] = instances.get_fields()[field].tolist()
        if field=="pred_classes":
            output[field] = instances.get_fields()[field].tolist()
        if field=="pred_boxes":
            output[field] = instances.get_fields()[field].tensor.tolist()
#         if field=="pred_masks":
#             output[field] = instances.get_fields()[field].tolist()

    output['image_size'] = instances.image_size

    output = json.dumps(output)
    
    return output

    
    
if __name__ == "__main__":
    """
    Test method which serializes Detectron2 predictions to JSON and back.
    """
    IMAGE = cv2.imread("coco_sample.jpg")
    CFG = get_cfg()
    CFG.merge_from_file("../R101-FPN/mask_rcnn_R_101_FPN_3x.yaml")
    CFG.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    CFG.MODEL.WEIGHTS = "../R101-FPN/R-101.pkl"
    PREDICTOR = DefaultPredictor(CFG)
    PREDICTIONS = PREDICTOR(IMAGE)
    DEVICE='cuda:0'
    
    pred = d2_to_json(PREDICTIONS)
    inst = json_to_d2(pred, DEVICE)


    
    
    