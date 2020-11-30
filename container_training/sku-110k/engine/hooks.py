"""Implementation of Hooks to be used in training loop"""
import torch
from detectron2.engine.hooks import HookBase
from detectron2.data import (
    # build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.utils import comm


class ValidationLoss(HookBase):
    def __init__(self, cfg, val_augmentation, period: int = 40):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(
            build_detection_train_loader(
                self.cfg,
                mapper=DatasetMapper(
                    self.cfg, is_train=True, augmentations=val_augmentation
                ),
            )
        )
        self._period = period
        self.num_steps = 0

    def after_step(self):
        self.num_steps += 1
        if self.num_steps % self._period == 0:
            data = next(self._loader)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                loss_dict = self.trainer.model(data)

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {
                    "val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
                }
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(
                        total_val_loss=losses_reduced, **loss_dict_reduced
                    )
                comm.synchronize()
        else:
            pass
