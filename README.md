# Goals
This repository implements port of latest [Detectron2](https://github.com/facebookresearch/detectron2/) ("D2") to [Amazon Sagemaker](https://aws.amazon.com/sagemaker/). Scope includes:
- [x] training of D2 on custom small balloon dataset using Sagemaker distributed training;
- [ ] training of D2 on COCO2017 using Sagemaker distributed training;
- [ ] deploying trained D2 model on Sagemaker Inference endpoint.


## Training on Balloon dataset.
This is a toy example based on [D2 tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=UkNbUzUOLYf0). To run training do following:
- update `container_training/Dockerfile` with `ENV SAGEMAKER_PROGRAM train_balloon.py`;
- execute `build_and_push.sh <image name>` to build a new customer Sagemaker-compatible container;
- open notebook `d2_byo_container_balloon_training.ipynb` and execute cells one by one.

As a result, you'll see that a Sagemaker training job will be executed within ~6 minutes: training will take ~1-2mins, the rest of the time will be spend on spinning up and tearing down the training.

## Training on COCO2017 dataset.
Work in progress. Instructions below are subject to change. To run first version of Coco2017 training do following:
- update `container_training/Dockerfile` with `ENV SAGEMAKER_PROGRAM train_coco.py`;
- open notebook `d2_byo_container_coco2017_training.ipynb` and execute cells one by one.

## Deploying trained D2 model for inference.
Work in progress




