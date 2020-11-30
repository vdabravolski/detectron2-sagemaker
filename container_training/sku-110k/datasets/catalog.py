"""Manage Grocery Store dataset"""
from typing import Sequence, Mapping
from pathlib import Path
import json
from functools import partial
from dataclasses import dataclass

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog, Metadata


@dataclass
class DataSetMeta:
    """
    Dataset metadata

    Attributes:
        name (str): dataset name
        classes (Sequence[str]): class of objects to detect
    """

    name: str
    classes: Sequence[str]

    def __str__(self):
        """Print dataset name and class names"""
        return (
            f"The object detection dataset {self.name} "
            f"can detect {len(self.classes)} type(s) of objects: "
            f"{self.classes}"
        )


def remove_dataset(ds_name: str):
    """Remove a previously registered data store """
    for channel in ("training", "validation"):
        DatasetCatalog.remove(f"{ds_name}_{channel}")


def aws_file_mode(path_imgs: str, path_annotation: str, label_name: str) -> Sequence[Mapping]:
    """
    Function that add dataset to Detectron by using the schema used by AWS for object detection in
    File Mode

    Args:
        path_imgs (str): path to folder that contains the images
        path_annotation (str): path to folder that contains the annotations
        label_name (str): label name used for object detection GT job

    Returns:
        Sequence[Mapping]: list of annotations

    Raises:
        FileNotFoundError: if the annotation channel misses some annotations for the image channel
    """
    dataset_dicts = []

    for img_id, p_image in enumerate(Path(path_imgs).iterdir()):
        if p_image.suffix.lower() not in (".png", ".jpg"):
            print(f"{p_image} is not an image and it will be ignore")
            continue

        json_file = f"{p_image.parts[-1].split('.')[0]}.json"
        annotation_file = Path(path_annotation) / json_file
        if not annotation_file.exists():
            raise FileNotFoundError(
                f"Broken dataset {p_image} has not a corresponding annotation file"
            )
        with open(str(annotation_file), "r") as fid:
            annotation = json.load(fid)

            record = {
                "file_name": str(p_image),
                "height": annotation[label_name]["image_size"][0]["height"],
                "width": annotation[label_name]["image_size"][0]["width"],
                "image_id": img_id,
            }

            objs = []

            for bbox in annotation[label_name]["annotations"]:
                objs.append(
                    {
                        "bbox": [
                            bbox["left"],
                            bbox["top"],
                            bbox["width"],
                            bbox["height"],
                        ],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": bbox["class_id"],
                    }
                )
            record["annotations"] = objs

            dataset_dicts.append(record)

    return dataset_dicts


def register_dataset(
    metadata: DataSetMeta,
    label_name: str,
    training: str,
    train_ann: str,
    validation: str,
    valid_ann: str,
) -> Metadata:
    """
    Register a training dataset to detectron2

    Args:
        metadata (DataSetMeta): metadata of the datasets to register
        label_name (str): label name used for object detection GT job
        training (str): path to training images
        train_ann (str): path to training annotations
        validation (str): path to validation images
        valid_ann (str): path to validation annotations

    Returns:
        Metadata: Detectron2 metadata file
    """
    channels = {
        "training": (training, train_ann),
        "validation": (validation, valid_ann),
    }

    for channel, datasets in channels.items():
        detectron_ds_name = f"{metadata.name}_{channel}"
        DatasetCatalog.register(
            detectron_ds_name, partial(aws_file_mode, datasets[0], datasets[1], label_name)
        )
        MetadataCatalog.get(detectron_ds_name).set(thing_classes=metadata.classes)
    return MetadataCatalog.get(f"{metadata.name}_training")
