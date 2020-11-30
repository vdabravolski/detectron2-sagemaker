"""Implementation of custom trainer"""
import os

from detectron2.engine import DefaultTrainer
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.config import CfgNode
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.utils import comm
from detectron2.engine import hooks
from detectron2.utils.events import CommonMetricPrinter

from engine.hooks import ValidationLoss

########################################################
# MACROs that define training and validation transforms
#! this shoud be passed as argument on the final dataset
########################################################

TRAIN_TRANSF = [
    T.ResizeShortestEdge(
        short_edge_length=(800,),
        max_size=1333,
        sample_style="choice",
    ),
    # T.RandomFlip(),
    # T.RandomApply(
    #     transform=T.RandomBrightness(intensity_min=0.75, intensity_max=1.25), prob=0.50
    # ),
    # T.RandomApply(
    #     transform=T.RandomContrast(intensity_min=0.76, intensity_max=1.25), prob=0.50
    # ),
    # T.RandomApply(
    #     transform=T.RandomSaturation(intensity_min=0.75, intensity_max=1.25), prob=0.50
    # ),
]
VAL_TRANSF = [
    T.ResizeShortestEdge(short_edge_length=(800,), max_size=1333, sample_style="choice")
]

########################################################
########################################################


class Trainer(DefaultTrainer):
    """
    Use a custom trainer. The main differences compared with DefaultTrainer are:

    * Use custom transforms rather than default ones defined in the config
    * Implement the build_evaluator to run evaluation on the validation datasets
    """

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name: str):
        return build_detection_test_loader(
            cfg,
            dataset_name,
            # pylint:disable=redundant-keyword-arg,missing-kwoa
            mapper=DatasetMapper(cfg, is_train=False, augmentations=VAL_TRANSF),
        )

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(
            cfg,
            # pylint:disable=redundant-keyword-arg,missing-kwoa
            mapper=DatasetMapper(cfg, is_train=True, augmentations=TRAIN_TRANSF),
        )

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        ret.append(ValidationLoss(cfg, VAL_TRANSF, cfg.VAL_LOG_PERIOD))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.VAL_LOG_PERIOD))
        return ret

    def build_writers(self):
        return [
            CommonMetricPrinter(self.cfg.SOLVER.MAX_ITER),
        ]
