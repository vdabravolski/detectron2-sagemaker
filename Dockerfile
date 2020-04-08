# Build an image of Detectron2 that can do 
# distributing training and inference in Amazon Sagemaker

# using Sagemaker PyTorch container as base image
FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.4.0-gpu-py36-cu101-ubuntu16.04
LABEL author="vadimd@amazon.com"

############# Installing latest builds ############

# This is to fix issue: https://github.com/pytorch/vision/issues/1489
# TODO: test if this works, otherwise, fall back to preinstalled torch/torchvision in SM container
RUN pip install --upgrade --force-reinstall torch torchvision cython

############# D2 section ##############

# installing dependecies for D2 https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

ENV FORCE_CUDA="1"
# Build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta" 

# Build D2 from latest sources
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"
# set location of training datasetm, Sagemaker containers copy all data from S3 to /opt/ml/input/data/{channels}
ENV DETECTRON2_DATASETS="/opt/ml/input/data/training"

############# SageMaker section ##############

COPY container_training /opt/ml/code
WORKDIR /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM train_coco.py

WORKDIR /

# Starts framework distributed framework
ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]