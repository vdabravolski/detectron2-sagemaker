from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data.catalog import MetadataCatalog


def d2_visualizer(pred_image, predictions, dataset_name="airbus_train"):
    
    if predictions is None:
        return
    
    im = cv2.imread(pred_image)
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get(dataset_name), 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels

    )
    v = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    return v.get_image()[:, :, ::-1]