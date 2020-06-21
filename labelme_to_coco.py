"""
This script converts XML files from Labelme tool to COCO format

LabelMe semantic format (original):
    - each image file has individual GT XML file;
    - each object in image has following polygon definition:
_________image.XML__________        
        <annotation>
            ...
            <object>
                <name>paved-area</name>
                <deleted>0</deleted>
                <verified>0</verified>
                <attributes/>
                <parts><hasparts/><ispartof>3</ispartof></parts>
                <date>10-Jun-2018 18:21:11</date>
                <id>0</id>
                <polygon>
                    <username>AWS08facce93177d07c6b68</username>
                    <pt><x>6</x><y>1</y></pt><pt><x>3383</x><y>0</y></pt><pt><x>5993</x><y>3</y></pt><pt><x>5996</x><y>3996</y></pt><pt><x>3</x><y>3996</y></pt>
                </polygon>
            </object>

COCO Semantic Segmentation format (target):

Refer to this article for description: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch#:~:text=The%20COCO%20dataset%20is%20formatted,%E2%80%9D%20(in%20one%20case)

________annotations.json_____
    {
    "info": {...},
    "licenses": [...],
    "images": [
        {
            "license": 4,
            "file_name": "000000397133.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
            "height": 427,
            "width": 640,
            "date_captured": "2013-11-14 17:02:52",
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
            "id": 397133
        },
        ...
    ],
    "annotations": 
        [
            {
                "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]], #  (x, y pixel positions)
                "area": 702.1057499999998,
                "iscrowd": 0, # whether it's a single object or not
                "image_id": 289343,  
                "bbox": [473.07,395.93,38.65,28.67], # [top left x position, top left y position, width, height]
                "category_id": 18, # category id, 18=dog
                "id": 1768
            },
            ...
        ]
    "categories": [
            {"supercategory": "person","id": 1,"name": "person"},
            ...
    ],
    "segment_info": [...] <-- Only in Panoptic annotations
    }
"""

import argparse
from lxml import etree
import os
import numpy as np
from PIL import Image
import json
from collections import namedtuple


# Defaults and assumptions around dataset structure
DEFAULT_CLASS_LIST = ["person"]
IMAGE_DIR = "images"
GT_DIR = "gt/semantic/label_me_xml"

DATASET_INFO = {
    "description": "Semantic Drone Dataset",
    "url": "https://www.tugraz.at/index.php?id=22387"
}

# Following approach from Cityscapes Scripts: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class
    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.
    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!
    'category'    , # The name of the category that this label belongs to
    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.
    'hasInstances', # Whether this label distinguishes between single instances or not
    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not
    'color'       , # The color of this label
    ] )

# By default provide only single class: person
DRONE_LABELS = [
    #       name                   id    trainId   category            catId     hasInstances    ignoreInEval   color
    Label(  'person'             ,  1,      0 ,     'human'            , 0       , True         , False        , (255, 22, 96) ),
]

# global counter of annotations
# to ensure that all annotations
# has global unique id
annotation_counter = 0


def _labels_to_categories(label_tuple):
    """
    Convert named tuple with labels/categories into
    COCO-formatted categories
    """

    categories = []

    for label in label_tuple:
        category = {
            "supercategory": label.category,
            "id": label.id,
            "name": label.name
        }
        categories.append(category)

    return categories


def _get_category_id(label_name, label_tuple):
    """
    Get category id based on label name
    """

    for label_record in label_tuple:
        if label_record.name.lower() == label_name.lower():
            return label_record.id
    assert False, f"Label {label_name} was not found."


def _get_files(dataset_dir):
    """
    Returns reference images and corresponding semantic GT annotations.
    Used to construct combined COCO-style annotation JSON.
    """

    files = []

    image_dir = os.path.join(dataset_dir, IMAGE_DIR)

    for dirpath, dirnames, imagenames in os.walk(image_dir):
        for image in imagenames:

            # get filename without extension
            image_filename = os.path.splitext(image)[0]

            files.append(
                (
                    os.path.join(dirpath, image),  # path to image
                    os.path.join(dataset_dir, GT_DIR, image_filename+".xml")  # construct path to XML file
                )
            )

    assert len(files), f"Didn't find any images in {dataset_dir}"

    return files


def _parse_file(image_file, xml_file, classes=DEFAULT_CLASS_LIST):
    """
    Takes Label Me XML file and return a COCO-style dict objects:
        - single image record;
        - associated with image annotations.
    """

    image_record = {}  # single dict
    annotation_records = []  # list of dict

    # Construct image record
    im = Image.open(image_file)
    width, height = im.size
    image_id = int(os.path.splitext(os.path.basename(image_file))[0])

    image_record = {
        "file_name": os.path.basename(image_file),
        "id": image_id,
        "height": height,
        "width": width
    }

    # Construct annotation list of dicts
    try:
        xml_doc = etree.parse(xml_file)
    except Exception as e:
        print(f"File {xml_file} is not parseable. Skipping...")
        print(e)
        return None, None

    # iterate over all defined object classes
    for obj_name in classes:

        # find all objects of the given class
        # get parent element of element=name with text attribute obj_name
        objects = xml_doc.xpath(f"//object/name[text()='{obj_name}']/..") 
        
        # iterate over all found objects of the given class
        for xml_obj in objects:
            
            annotation = {} # record for each found annotation of given object class

            px = xml_obj.xpath(".//x/text()")  # get all X coords
            py = xml_obj.xpath(".//y/text()")  # get all Y coords

            # convert string vertices to float
            px = np.array(px).astype(np.float)
            py = np.array(py).astype(np.float)

            import itertools

            # BBOX format: [top left x position, top left y position, width, height]
            xmin = np.min(px)
            ymin = np.min(py)
            width = np.max(px) - xmin
            height = np.max(py) - ymin
            annotation["bbox"] = [xmin, ymin, width, height]

            # Needed for Detectron2
            # See: https://detectron2.readthedocs.io/_modules/detectron2/structures/boxes.html 
            annotation["bbox_mode"] = 1
            annotation["segmentation"] = [list(itertools.chain.from_iterable(zip(px, py)))]
            annotation["category_id"] = _get_category_id(obj_name, DRONE_LABELS)
            annotation["image_id"] = image_id

            # assign unique id
            global annotation_counter
            annotation["id"] = annotation_counter
            annotation_counter += 1

            annotation_records.append(annotation)

    return image_record, annotation_records


def main(args):

    files = _get_files(args.dataset_dir)

    coco_json = {}  # COCO-compatible JSON

    coco_json["info"] = DATASET_INFO
    coco_json["categories"] = [] # TODO: this needs to be fixed
    coco_json["images"] = []
    coco_json["annotations"] = []
    coco_json["categories"] = _labels_to_categories(DRONE_LABELS)

    for image, gt_xml in files:
        print(f"processing file {gt_xml}")
        image_record, annotation_records = _parse_file(image, gt_xml)

        if image_record is not None:  # to handle when XML is not parseable
            coco_json["images"].append(image_record)
            coco_json["annotations"] += annotation_records

    with open(os.path.join(args.output_dir, "train.json"), 'w') as f:
        print(f"Writing results in {f}")
        json.dump(coco_json, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, help="path to folder with GT images")
    parser.add_argument('--output-dir', type=str, default=".", help="where to store output JSON")
    parser.add_argument('--object-names', type=float,
                        default=DEFAULT_CLASS_LIST, help="Define how many files will be used to concatenate train file. Choose between 0.0 and 1.0")
    args = parser.parse_args()

    main(args)
