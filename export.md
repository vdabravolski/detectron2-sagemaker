1. When trying to run scripting of backbone network, face issue with unsupported operation:
sample command: 
`python torchscript_export.py --config ./trained_models/R50-C4/faster_rcnn_R_50_C4_1x.yaml --image ./trained_models/model_R_50_FPN_1x/coco_sample.jpg --task script --weights ./trained_models/R50-C4/R50-C4.pkl` 


2. When trying to run tracing, face the issue that backbone network returns dict, and not tensor. Only tensor output is supported in Torch 1.5
Sample command:
`python torchscript_export.py --config ./trained_models/R50-C4/faster_rcnn_R_50_C4_1x.yaml --image ./trained_models/model_R_50_FPN_1x/coco_sample.jpg --task trace --weights ./trained_models/R50-C4/R50-C4.pkl`

However, recently a new feature was added in torch master to support dict outputs: https://github.com/pytorch/pytorch/issues/27743 Need to recompile torch from source and retest.