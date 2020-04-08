#!/usr/bin/env python
"""
This is a re-implementation of Detectron2 sample training script: https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py
"""

# import some common libraries
import argparse
import sys
import logging
import os
from collections import OrderedDict
import torch
import json
    
# import some common detectron2 utilities
# TODO: check imports and remove redundant
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.utils.logger import setup_logger
setup_logger() # D2 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        elif evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def main(*params):
    
    cfg = params[0]
    args = params[1]
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def custom_argument_parser(config_file, num_gpus, num_machines, machine_rank, master_addr,master_port):
    """
    Create a parser with some common arguments used by detectron2 users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Detectron2 Training")    
    parser.add_argument("--config-file", default=config_file, metavar="FILE", help="path to config file")
    parser.add_argument("--resume", action="store_true",help="whether to attempt to resume from the checkpoint directory",)
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=num_gpus, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=num_machines)
    parser.add_argument("--machine-rank", type=int, default=machine_rank, help="the rank of this machine (unique per machine)")

    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14 # replace D2 port allocation with predefined Sagemaker distributed cluster params
    parser.add_argument("--dist-url", default="tcp://{}:{}".format(master_addr, master_port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser



if __name__ == "__main__":
    
    # Sagemaker configuration
    print('Starting training...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default="nccl", help='backend for distributed operations.')
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument('--current-host', type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument('--model-dir', type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--num-gpus', type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument('--num-cpus', type=int, default=os.environ["SM_NUM_CPUS"])
    parser.add_argument('--config-file', default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", metavar="FILE", help="path to config file")
    sm_args = parser.parse_args()
    
    # Derive parameters of distributed training
    number_of_processes = sm_args.num_gpus if sm_args.num_gpus > 0 else sm_args.num_cpus
    number_of_machines = len(sm_args.hosts)
    world_size = number_of_processes * number_of_machines
    logger.info('Running \'{}\' backend on {} nodes and {} processes. World size is {}.'.
                format(sm_args.backend, number_of_machines, number_of_processes, world_size))
    machine_rank = sm_args.hosts.index(sm_args.current_host)
    master_addr = sm_args.hosts[0]
    master_port = '55555'
    
    # D2 configuration
    # See for details: https://detectron2.readthedocs.io/modules/config.html#config-references
    # TODO: Training call - configs are designed for 8 gpus, so may need to troubleshoot
    # ./train_net.py --num-gpus 8 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
    
    #D2 expects ArgParser object to configure Trainer. As distributed config is derived from Sagemaker job, we are constructing artificial ArgParse object here. TODO: fix it.
    config_file_path = f"{os.environ['SM_MODULE_DIR']}/detectron2/configs/{sm_args.config_file}"
    print(config_file_path)
    d2_args = custom_argument_parser(config_file_path, sm_args.num_gpus,
                                     number_of_machines, machine_rank, master_addr,master_port).parse_args() 
    # TODO: need to update some arguments from here: https://github.com/facebookresearch/detectron2/blob/cd9ac61861e83856ed8854c98ebaf383b77950ae/detectron2/engine/defaults.py#L49
    
    # implements this logic https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py#L114-L123
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(sm_args.config_file))
    cfg.DATALOADER.NUM_WORKERS = 2 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(sm_args.config_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = world_size # number ims_per_batch should be divisible by number of workers. D2 assertion.
    cfg.SOLVER.BASE_LR = 0.00025  # TODO: check good LR, not clear how LR will depend on number of hosts/GPUs
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # TODO: need to confirm that this is correct # of classes for COCO2017
    cfg.OUTPUT_DIR = os.environ['SM_OUTPUT_DATA_DIR'] # TODO check that this config works fine
    cfg.freeze()
    
    default_setup(cfg, d2_args)
    
    # Launch D2 distributed training
    launch(
        main,
        sm_args.num_gpus,
        num_machines=number_of_machines,
        machine_rank=machine_rank,
        dist_url=f"tcp://{master_addr}:{master_port}",
        args=(cfg, d2_args,),
    )