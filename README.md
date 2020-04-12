This is not official AWS repository. Code provided "as is".

# Goals
This repository implements port of latest [Detectron2](https://github.com/facebookresearch/detectron2/) ("D2") to [Amazon Sagemaker](https://aws.amazon.com/sagemaker/). Scope includes:
- [x] training of D2 on custom small balloon dataset using Sagemaker distributed training;
- [ ] training of D2 on COCO2017 using Sagemaker distributed training;
- [ ] deploying trained D2 model on Sagemaker Inference endpoint.

## Containers
Amazon Sagemaker uses docker containers both for training and inference. For Detectron2 it's planned to create two separate containers for training and serving.

**Note**: by default training container compiles Detectron2 for Volta architecture (Tesla V100 GPUs). If you'd like to run training on other GPU architectures, consider updating this [environment variable](https://github.com/vdabravolski/detectron2-sagemaker/blob/e6252211b819815962207d1a61d5675d213e0f25/Dockerfile#L21). Here is an example on how to compile Detectron2 for all supported architectures:

`ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"`

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
**Work in progress**: As of April'2020, Detectron2 supports model export to [Caffe2 format](https://detectron2.readthedocs.io/tutorials/deployment.html#caffe2-deployment) (via ONNX) for serving. The intention is to deploy D2 models for serving as [Amazon Elastic Inference](https://aws.amazon.com/machine-learning/elastic-inference/) ("EI") for cost efficiencies, and currently EI only support PyTorch, MxNet, and Tensorflow runtime environment. Hence, following options will be considered:
- Create custom container for Caffe2 serving in Sagemaker;
- Convert D2 model to ONXX. However, D2 has some [custom ops](https://github.com/facebookresearch/detectron2/blob/2ca36e3cbfb2c84c18502221564b629f3877e8be/detectron2/export/api.py#L63-L66) not supported by other runtimes. Will likely need to customize it further.






