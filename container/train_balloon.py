# import some common libraries
import numpy as np
import cv2
import random
import argparse
import subprocess
import sys
import logging
    
# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, launch
from detectron2.utils.logger import setup_logger

# packages neededs for custom dataset
import os
import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog

# Logging TODO: remove duplicative loggers
setup_logger() # D2 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
    

def train(*args):

    prepare_dataset()

    # D2 configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args[0] # number ims_per_batch should be divisible by number of workers. D2 assertion.
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.OUTPUT_DIR = os.environ['SM_OUTPUT_DATA_DIR'] # TODO check that this config works fine

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

def prepare_dataset():
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(os.environ[f"SM_CHANNEL_{d.upper()}"]))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])


def get_balloon_dicts(img_dir):
    
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts



if __name__ == "__main__":
    # Sagemaker configuration
    print('Starting training...')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backend', type=str, default="nccl", help='backend for distributed operations.') # TODO: it looks like we are not passing backend. 
                                                                                                           # D2 defaults it to NCCL
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument('--current-host', type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument('--model-dir', type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--num-gpus', type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument('--num-cpus', type=int, default=os.environ["SM_NUM_CPUS"])
    args = parser.parse_args()

    number_of_processes = args.num_gpus if args.num_gpus > 0 else args.num_cpus
    number_of_machines = len(args.hosts)
    world_size = number_of_processes * number_of_machines
    logger.info('Running \'{}\' backend on {} nodes and {} processes. World size is {}.'.format(
        args.backend, number_of_machines, number_of_processes, world_size
    ))
    machine_rank = args.hosts.index(args.current_host)
    master_addr = args.hosts[0]
    master_port = '55555'
    
    #TODO: delete debug section
    print(f"machine_rank:{machine_rank}")
    print(f"master_addr:{master_addr}")
    print(f"master_port:{master_port}")
    print(f"num_gpus:{args.num_gpus}")
    
    # Launch D2 distributed training
    launch(
        train,
        args.num_gpus,
        num_machines=number_of_machines,
        machine_rank=machine_rank,
        dist_url=f"tcp://{master_addr}:{master_port}",
        args=(world_size,),
    )