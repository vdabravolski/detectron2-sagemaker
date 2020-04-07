# Build an image of Detectron2 that can do 
# distributing training and inference in Amazon Sagemaker

# using Sagemaker PyTorch container as base image
FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.4.0-gpu-py36-cu101-ubuntu16.04
LABEL author="vadimd@amazon.com"

############# Installing latest builds ############

# This is to fix issue: https://github.com/pytorch/vision/issues/1489
# TODO: test if this works, otherwise, fall back to preinstalled torch/torchvision in SM container
RUN pip install --upgrade --force-reinstall torch torchvision

############# D2 section ##############

# installing dependecies for detectron2 https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'
# Build D2 from latest sources
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
ENV FORCE_CUDA="1"

# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# Set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"
ENV DETECTRON2_DATASETS="/opt/ml/input/data/training"

############# SageMaker section ##############

COPY container /opt/ml/code
WORKDIR /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM train.py

WORKDIR /

# Starts framework distributed framework
ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]