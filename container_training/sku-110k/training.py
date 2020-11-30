"""Entry point of the container to be used for SageMaker training of Detectron2 algorithms"""
import os
import argparse
import logging
import sys
import ast
import json
import shutil

from detectron2.engine import launch
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo

from datasets.catalog import register_dataset, DataSetMeta
from engine.custom_trainer import Trainer

##############
# Macros
##############
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


########################
# Implementation Details
########################


def _config_training(args: argparse.Namespace) -> CfgNode:
    """
    Create a configuration node from the script arguments. In this application we consider
    object detection use case only. We finetune object detection networks trained on COCO dataset
    to a custom use case

    Args:
        args (argparse.Namespace): script arguments

    Returns:
        CfgNode: configuration that is used by Detectron2 to train a model

    Raises:
        RuntimeError: if the combination of model_type, backbone, lr_schedule is not part of
            Detectron2's zoo. Please have a look at
            https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
            if this exception is thrown
    """
    cfg = get_cfg()
    pretrained_model = (
        f"COCO-Detection/{args.model_type}_{args.backbone}_{args.lr_schedule}x.yaml"
    )
    LOGGER.info(f"Loooking for the pretrained model {pretrained_model}...")
    try:
        cfg.merge_from_file(model_zoo.get_config_file(pretrained_model))
    except RuntimeError as err:
        LOGGER.error(f"{err}: check model backbone and lr schedule combination")
        raise
    cfg.DATASETS.TRAIN = (f"{args.dataset_name}_training",)
    cfg.DATASETS.TEST = (f"{args.dataset_name}_validation",)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pretrained_model)
    LOGGER.info(f"{pretrained_model} correctly loaded")

    cfg.SOLVER.CHECKPOINT_PERIOD = 20000
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.num_iter
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.num_rpn
    if args.model_type == "faster_rcnn":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(args.classes)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.pred_thr
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thr
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
        cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = args.reg_loss_type
        cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = args.bbox_reg_loss_weight
        cfg.MODEL.RPN.POSITIVE_FRACTION = args.bbox_rpn_pos_fraction
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = args.bbox_head_pos_fraction
    elif args.model_type == "retinanet":
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.pred_thr
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = args.nms_thr
        cfg.MODEL.RETINANET.NUM_CLASSES = len(args.classes)
        cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = args.reg_loss_type
        cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = args.focal_loss_gamma
        cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = args.focal_loss_alpha

        # TODO: add these options as parameters
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    else:
        assert False, f"Add implementation for model {args.model_type}"
    cfg.MODEL.DEVICE = "cuda" if args.num_gpus else "cpu"

    cfg.TEST.DETECTIONS_PER_IMAGE = args.det_per_img

    cfg.OUTPUT_DIR = args.model_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def _train_impl(args) -> None:
    """
    Training implementation executes the following steps:

    * Register the dataset to Detectron2 catalog
    * Create the configuration node for training
    * Run training
    * Serialize the training configuration to a JSON file, which is required for prediction
    """

    dataset = DataSetMeta(name=args.dataset_name, classes=args.classes)
    LOGGER.info(f"Dataset registered! {dataset}")
    register_dataset(
        metadata=dataset,
        label_name=args.label_name,
        training=args.training_channel,
        train_ann=args.training_annotation_channel,
        validation=args.validation_channel,
        valid_ann=args.validation_annotation_channel,
    )

    cfg = _config_training(args)

    cfg.setdefault("VAL_LOG_PERIOD", args.log_period)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)

    if cfg.MODEL.DEVICE != "cuda":
        err = RuntimeError("A CUDA device is required to launch training")
        LOGGER.error(err)
        raise err
    trainer.train()

    # Save only if in master process
    if args.current_host == args.hosts[0]:
        with open(f"{cfg.OUTPUT_DIR}/config.json", "w") as fid:
            json.dump(cfg, fid, indent=2)

        code_folder = f"{cfg.OUTPUT_DIR}/code/"
        os.makedirs(os.path.dirname(code_folder), exist_ok=True)
        shutil.copy("inference.py", code_folder)


##########
# Training
##########


def train(args: argparse.Namespace) -> None:
    """
    Training script uses Detectron2 wrapper 'launch' to simply distribute training

    Args:
        args (argparse.Namespace): please refer to argument helps for details (python $thisfile - h)
    """
    args.classes = ast.literal_eval(args.classes)

    machine_rank = args.hosts.index(args.current_host)
    LOGGER.info(f"Machine rank: {machine_rank}")
    master_addr = args.hosts[0]
    master_port = "55555"

    url = "auto" if len(args.hosts) == 1 else f"tcp://{master_addr}:{master_port}"
    LOGGER.info(f"Device URL: {url}")

    launch(
        _train_impl,
        num_gpus_per_machine=args.num_gpus, # // len(args.hosts),
        num_machines=len(args.hosts),
        dist_url=url,
        machine_rank=machine_rank,
        args=(args,),
    )


#############
# Script API
#############

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    # Pretrained model
    PARSER.add_argument(
        "--model-type",
        type=str,
        default="faster_rcnn",
        choices=["faster_rcnn", "retinanet"],
        metavar="MT",
        help=(
            "Type of architecture to be used for object detection; "
            "two options are supported: 'faster_rccn' and 'retinanet' "
            "(default: faster_rcnn)"
        ),
    )
    PARSER.add_argument(
        "--backbone",
        type=str,
        default="R_50_C4",
        choices=[
            "R_50_C4",
            "R_50_DC5",
            "R_50_FPN",
            "R_101_C4",
            "R_101_DC5",
            "R_101_FPN",
            "X_101_32x8d_FPN",
        ],
        metavar="B",
        help=(
            "Encoder backbone, how to read this field: "
            "R50 (RetinaNet-50), R100 (RetinaNet-100), X101 (ResNeXt-101); "
            "C4 (Use a ResNet conv4 backbone with conv5 head), "
            "DC5 (ResNet conv5 backbone with dilations in conv5) "
            "FPN (Use a FPN on top of resnet) ;"
            "Attention! Only some combinations are supported, please refer to the original doc "
            "(https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) "
            "(default: R_50_C4)"
        ),
    )
    PARSER.add_argument(
        "--lr-schedule",
        type=int,
        default=1,
        choices=[1, 3],
        metavar="LRS",
        help=(
            "Length of the training schedule, two values are supported: 1 or 3. "
            "1x = 16 images / it * 90,000 iterations in total with the LR reduced at 60k and 80k."
            "3x = 16 images / it * 270,000 iterations in total with the LR reduced at 210k and 250k"
            "(default: 1)"
        ),
    )
    # Hyper-parameters
    PARSER.add_argument(
        "--num-workers",
        type=int,
        default=2,
        metavar="NW",
        help="Number of workers used to by the data loader (default: 2)",
    )
    PARSER.add_argument(
        "--lr",
        type=float,
        default=0.00025,
        metavar="LR",
        help="Base learning rate value (default: 0.00025)",
    )
    PARSER.add_argument(
        "--num-iter",
        type=int,
        default=1000,
        metavar="I",
        help="Maximum number of iterations (default: 1000)",
    )
    PARSER.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="B",
        help="Number of images per batch across all machines (default: 16)",
    )
    PARSER.add_argument(
        "--num-rpn",
        type=int,
        default=100,
        metavar="R",
        help="Total number of RPN examples per image (default: 100)",
    )
    PARSER.add_argument(
        "--reg-loss-type",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "giou"],
        metavar="RLT",
        help=("Loss type used for regression subnet " "(default: smooth_l1)"),
    )

    # RetinaNet Specific
    PARSER.add_argument(
        "--focal-loss-gamma",
        type=float,
        default=2.0,
        metavar="FLG",
        help="Focal loss gamma, used in RetinaNet (default: 2.0)",
    )
    PARSER.add_argument(
        "--focal-loss-alpha",
        type=float,
        default=0.25,
        metavar="FLA",
        help="Focal loss alpha, used in RetinaNet. It must be in [0.1,1] (default: 0.25)",
    )

    # Faster-RCNN Specific
    PARSER.add_argument(
        "--bbox-reg-loss-weight",
        type=float,
        default=1.0,
        help="Weight regression loss (default: 0.1)",
    )
    PARSER.add_argument(
        "--bbox-rpn-pos-fraction",
        type=float,
        default=0.5,
        help="Target fraction of foreground (positive) examples per RPN minibatch (default: 0.5)",
    )
    PARSER.add_argument(
        "--bbox-head-pos-fraction",
        type=float,
        default=0.25,
        help="Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0) (default: 0.25)",
    )
    PARSER.add_argument(
        "--log-period",
        type=int,
        default=40,
        help="Occurence in number of iterations at which loss values are logged"
    )

    # Inference Parameters
    PARSER.add_argument(
        "--det-per-img",
        type=int,
        default=200,
        metavar="R",
        help="Maximum number of detections to return per image during inference (default: 200)",
    )
    PARSER.add_argument(
        "--nms-thr",
        type=float,
        default=0.5,
        metavar="NMS",
        help="If IoU is bigger than this value, only more confident pred is kept "
        "(default: 0.5)",
    )
    PARSER.add_argument(
        "--pred-thr",
        type=float,
        default=0.5,
        metavar="PT",
        help="Minimum confidence score to retain prediction (default: 0.5)",
    )

    # Mandatory parameters
    PARSER.add_argument(
        "--classes", type=str, metavar="C", help="List of classes of objects"
    )
    PARSER.add_argument(
        "--dataset-name", type=str, metavar="DS", help="Name of the dataset"
    )
    PARSER.add_argument(
        "--label-name",
        type=str,
        metavar="DS",
        help="Name of category of objects to detect (e.g. 'object')",
    )

    # Container Environment
    PARSER.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    PARSER.add_argument(
        "--training-channel",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"],
        help="Path folder that contains training images (File mode)",
    )
    PARSER.add_argument(
        "--training-annotation-channel",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING_ANNOTATION"],
        help="Path to folder that contains one JSON annotation file per training image (File mode)",
    )
    PARSER.add_argument(
        "--validation-channel",
        type=str,
        default=os.environ["SM_CHANNEL_VALIDATION"],
        help="Path folder that contains validation images (File mode)",
    )
    PARSER.add_argument(
        "--validation-annotation-channel",
        type=str,
        default=os.environ["SM_CHANNEL_VALIDATION_ANNOTATION"],
        help="Path to folder that contains one JSON annotation file per val image (File mode)",
    )

    PARSER.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    PARSER.add_argument(
        "--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"])
    )
    PARSER.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )

    train(PARSER.parse_args())
