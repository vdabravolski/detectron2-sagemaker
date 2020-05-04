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
import shutil
from torch.nn.parallel import DistributedDataParallel

    
# import some common detectron2 utilities
# TODO: check imports and remove redundant
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import  build_model
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)


from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.utils.logger import setup_logger
setup_logger() # D2 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _setup(sm_args):
    """
    Create D2 configs and perform basic setups.  
    """
    
    # D2 expects ArgParser.NameSpace object to ammend Cfg node.
    # We are constructing artificial ArgParse object here. TODO: consider refactoring it in future.
    
    # based on user input, selecting a correct base config file
    if sm_args.local_config_file != None:
        config_file_path = f"{os.environ['SM_MODULE_DIR']}/{sm_args.local_config_file}"
    else:
        config_file_path = f"{os.environ['SM_MODULE_DIR']}/detectron2/configs/{sm_args.config_file}"
        
    
    d2_args = _custom_argument_parser(config_file_path, sm_args.opts, sm_args.resume, sm_args.eval_only)
    
    cfg = get_cfg()
    cfg.merge_from_file(config_file_path) # get baseline parameters from YAML config
    list_opts = _opts_to_list(sm_args.opts) # convert training hyperparameters from SM format to D2
    cfg.merge_from_list(list_opts) # override defaults params from D2 config_file with user defined hyperparameters

    # Parameters below are hardcoded as they are specific to Sagemaker environment, no configuration needed.
    _, _ , world_size = _get_sm_world_size(sm_args)
    cfg.SOLVER.IMS_PER_BATCH = world_size # number ims_per_batch should be divisible by number of workers. D2 assertion. TODO: currently equal to world_size
    cfg.OUTPUT_DIR = os.environ['SM_OUTPUT_DATA_DIR'] # TODO check that this config works fine
    cfg.freeze()
    
    default_setup(cfg, d2_args)
    
    return cfg
    

def _custom_argument_parser(config_file, opts, resume, eval_only):
    """
    Create a parser with some common arguments for Detectron2 training script.
    Returns:
        argparse.NameSpace:
    """
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument("--config-file", default=None, metavar="FILE", help="path to config file")
    parser.add_argument("--opts",default=None ,help="Modify config options using the command-line")
    parser.add_argument("--resume", type=str, default="True", help="whether to attempt to resume from the checkpoint directory",)
    parser.add_argument("--eval-only", type=str, default="False", help="perform evaluation only")
    
    args = parser.parse_args(["--config-file", config_file,
                             "--resume", resume,
                             "--eval-only", eval_only,
                             "--opts", opts])
    return args

def _opts_to_list(opts):
    """
    This function takes a string and converts it to list of string params (YACS expected format). 
    E.g.:
        ['SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.9999'] -> ['SOLVER.IMS_PER_BATCH', '2', 'SOLVER.BASE_LR', '0.9999']
    """
    import re
    
    if opts!=None:
        list_opts = re.split('\s+', opts)
        return list_opts
    return ""


def _get_sm_world_size(sm_args):
    """
    Calculates number of devices in Sagemaker distributed cluster
    """
    
    number_of_processes = sm_args.num_gpus if sm_args.num_gpus > 0 else sm_args.num_cpus
    number_of_machines = len(sm_args.hosts)
    world_size = number_of_processes * number_of_machines
    
    return number_of_processes, number_of_machines, world_size


def _save_model(model, model_dir=os.environ['SM_MODEL_DIR']):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model_final.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)
    
    # copy config.yaml to model dir
    config_path = os.path.join(os.environ['SM_OUTPUT_DATA_DIR'], "config.yaml")
    new_config_path = os.path.join(model_dir, "config.yaml")
    shutil.copyfile(config_path, new_config_path)

    try:
        # copy checkpoint file to model dir
        checkpoint_path = os.path.join(os.environ['SM_OUTPUT_DATA_DIR'], "last_checkpoint")
        new_checkpoint_path = os.path.join(model_dir, "last_checkpoint")
        shutil.copyfile(checkpoint_path, new_checkpoint_path)
    except:
        logger.debug("D2 checkpoint file is not available.")

        
def get_evaluator(cfg, dataset_name, output_folder=None):
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
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

        
def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            

def main(sm_args):
    
    cfg = _setup(sm_args)
    
    model = build_model(cfg)
    
    # Converting string params to boolean flags as Sagemaker doesn't support currently boolean flags as hyperparameters.
    eval_only = True if sm_args.eval_only=="True" else False
    resume = True if sm_args.resume=="True" else False
    
    if eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=resume)
    _save_model(model)
    
    return do_test(cfg, model)
    

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
    parser.add_argument('--resume', type=str, default="True")
    parser.add_argument('--eval-only', type=str, default="False")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config-file', type=str, default=None, metavar="FILE", help="If config file specificed, then one of the Detectron2 configs will be used. \
                       Refer to https://github.com/facebookresearch/detectron2/tree/master/configs")
    group.add_argument('--local-config-file', type=str, default=None, metavar="FILE", help="If local config file specified, then config file \
                       from container_training directory will be used.")
    parser.add_argument('--opts', default=None)
    sm_args = parser.parse_args()
    
    # Derive parameters of distributed training
    number_of_processes, number_of_machines, world_size = _get_sm_world_size(sm_args)
    logger.info('Running \'{}\' backend on {} nodes and {} processes. World size is {}.'.
                format(sm_args.backend, number_of_machines, number_of_processes, world_size))
    machine_rank = sm_args.hosts.index(sm_args.current_host)
    master_addr = sm_args.hosts[0]
    master_port = '55555'
        
    # Launch D2 distributed training
    launch(
        main,
        num_gpus_per_machine=number_of_processes,
        num_machines=number_of_machines,
        machine_rank=machine_rank,
        dist_url=f"tcp://{master_addr}:{master_port}",
        args=(sm_args,),
    )