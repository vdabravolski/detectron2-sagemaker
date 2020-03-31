# Build an image of Detectron2 that can do 
# distributing training and inference in Amazon Sagemaker

# using Sagemaker PyTorch container as base image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.4.0-gpu-py36-cu101-ubuntu16.04

LABEL author="vadimd@amazon.com"

COPY container /opt/ml/code
WORKDIR /opt/ml/code
RUN chmod +x serve
RUN chmod +x train

# installing dependecies for detectron2 https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile
RUN pip install --user torch torchvision tensorboard cython
RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2.git /opt/ml/code/detectron2
ENV FORCE_CUDA="1"

# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# Installing Detectron2 as library. You have two options:
# option 1: rebuild from scratch
RUN pip install --user -e /opt/ml/code/detectron2
# option 2: install pre-built. It's faster, but not latest version and may be mismatch with CUDA env.
#RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html

# Set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"
ENV DETECTRON2_DATASETS="/opt/ml/data/training"
ENV SAGEMAKER_PROGRAM train
WORKDIR /