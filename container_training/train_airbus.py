import argparse
import sys
import logging
import os
import torch
import json
import shutil
from torch.nn.parallel import DistributedDataParallel
from os import walk

# import some common detectron2 utilities
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer, default_argument_parser,\
    default_setup, hooks, launch

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer


from detectron2.utils.logger import setup_logger
setup_logger()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _register_dataset(dataset_name):
    from detectron2.data.datasets import register_coco_instances

    dataset_location = os.environ["DETECTRON2_DATASETS"]
    annotation_file = "train.json"
    image_dir = "ships_train2018"

    register_coco_instances(dataset_name, {}, os.path.join(dataset_location, annotation_file), 
                            os.path.join(dataset_location, image_dir))

    drone_meta = MetadataCatalog.get(dataset_name)
    logger.info(f"Registered dataset {dataset_name}")
    logger.info(drone_meta)


def _setup(sm_args):
    """
    Create D2 configs and perform basic setups.  
    """

    # Choose whether to use config file from D2 model zoo or 
    # user supplied config file ("local_config_file")
    if sm_args.local_config_file is not None:
        config_file_path = f"{os.environ['SAGEMAKER_SUBMIT_DIRECTORY']}/{sm_args.local_config_file}"
        config_file = sm_args.local_config_file
    else:
        config_file_path = f"{os.environ['SAGEMAKER_SUBMIT_DIRECTORY']}/detectron2/configs/{sm_args.config_file}"
        config_file = sm_args.config_file

    # Register custom dataset
    dataset_name = "airbus"
    _register_dataset(dataset_name)
    
    # Build config file
    cfg = get_cfg() # retrieve baseline config: https://github.com/facebookresearch/detectron2/blob/master/detectron2/config/defaults.py
    cfg.merge_from_file(config_file_path) # merge defaults with provided config file
    list_opts = _opts_to_list(sm_args.opts)
    cfg.merge_from_list(list_opts) # override parameters with user defined opts
    cfg.DATASETS.TRAIN = (dataset_name,) # define dataset used for training
    cfg.DATASETS.TEST = ()  # no test dataset available
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.OUTPUT_DIR = os.environ['SM_OUTPUT_DATA_DIR']
    cfg.freeze()
    
    # D2 expects ArgParser.NameSpace object to ammend Cfg node.
    d2_args = _custom_argument_parser(config_file_path, sm_args.opts, sm_args.resume)
    # Perform training setup before training job starts
    default_setup(cfg, d2_args)
    
    return cfg
    

def _custom_argument_parser(config_file, opts, resume):
    """
    Create a parser with some common arguments for Detectron2 training script.
    Returns:
        argparse.NameSpace:
    """
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument("--config-file", default=None, metavar="FILE", help="path to config file")
    parser.add_argument("--opts",default=None ,help="Modify config options using the command-line")
    parser.add_argument("--resume", type=str, default="True", help="whether to attempt to resume from the checkpoint directory",)
    
    args = parser.parse_args(["--config-file", config_file,
                             "--resume", resume,
                             "--opts", opts])
    return args


def _opts_to_list(opts):
    """
    This function takes a string and converts it to list of string params (YACS expected format). 
    E.g.:
        ['SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.9999'] -> ['SOLVER.IMS_PER_BATCH', '2', 'SOLVER.BASE_LR', '0.9999']
    """
    import re
    
    if opts is not None:
        list_opts = re.split('\s+', opts)
        return list_opts
    return ""


def get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """

    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555" # port is defined by Sagemaker
    world["is_master"] = current_host == sorted(hosts)[0]

    return world

def _save_model():
    """
    This method copies model weight, config, and checkpoint(optionally)
    from output directory to model directory.
    Sagemaker then automatically archives content of model directory
    and adds it to model registry once training job is completed.
    """
    
    logger.info("Saving the model into model dir")
    
    model_dir = os.environ['SM_MODEL_DIR']
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    
    # copy model_final.pth to model dir
    model_path = os.path.join(output_dir, "model_final.pth")
    new_model_path = os.path.join(model_dir, 'model_final.pth')
    shutil.copyfile(model_path, new_model_path)

    # copy config.yaml to model dir
    config_path = os.path.join(output_dir, "config.yaml")
    new_config_path = os.path.join(model_dir, "config.yaml")
    shutil.copyfile(config_path, new_config_path)

    try:
        # copy checkpoint file to model dir
        checkpoint_path = os.path.join(output_dir, "last_checkpoint")
        new_checkpoint_path = os.path.join(model_dir, "last_checkpoint")
        shutil.copyfile(checkpoint_path, new_checkpoint_path)
    except Exception:
        logger.debug("D2 checkpoint file is not available.")


def main(sm_args, world):
    
    cfg = _setup(sm_args)
    
    is_zero_rank = comm.get_local_rank()==0
    
    trainer = DefaultTrainer(cfg)
    resume = True if sm_args.resume == "True" else False
    trainer.resume_or_load(resume=resume)
    trainer.train()
    
    if world["is_master"] and is_zero_rank:
        _save_model()
    


if __name__ == "__main__":
    
    # Sagemaker configuration
    logger.info('Starting training...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default="True") # TODO: is it relevant?
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config-file', type=str, default=None, metavar="FILE", help="If config file specificed, then one of the Detectron2 configs will be used. \
                       Refer to https://github.com/facebookresearch/detectron2/tree/master/configs")
    group.add_argument('--local-config-file', type=str, default=None, metavar="FILE", help="If local config file specified, then config file \
                       from container_training directory will be used.")
    parser.add_argument('--opts', default=None)
    sm_args = parser.parse_args()
    
    try:
        f = []
        for (dirpath, dirnames, filenames) in walk("/"):
            f.extend(dirnames)
            break        
        print(f"top level dirs:{f}")
        
        f = []
        for (dirpath, dirnames, filenames) in walk("/fsx"):
            f.extend(dirnames)
            break
        print(f"mount point dirs:{f}")
    except Exception as e:
        print(e)
    
    # Derive parameters of distributed training
    world = get_training_world()
    logger.info(f'Running "nccl" backend on {world["number_of_machines"]} machine(s)'
                f'each with {world["number_of_processes"]} GPU device(s). World size is'
                f'{world["size"]}. Current machine rank is {world["machine_rank"]}.')
        
    # Launch D2 distributed training
    launch(
        main,
        num_gpus_per_machine=world["number_of_processes"],
        num_machines=world["number_of_machines"],
        machine_rank=world["machine_rank"],
        dist_url=f"tcp://{world['master_addr']}:{world['master_port']}",
        args=(sm_args, world,),
    )