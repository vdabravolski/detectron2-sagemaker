This is not official AWS repository. Code provided "as is".

# Goals
This repository implements port of latest [Detectron2](https://github.com/facebookresearch/detectron2/) ("D2") to [Amazon Sagemaker](https://aws.amazon.com/sagemaker/). Scope includes:
- [x] train Detectron2 models on COCO2017 using Sagemaker distributed training;
- [x] deploy trained D2 model on Sagemaker Inference endpoint.
- [x] finetune D2 model on custom dataset using Sagemaker distributed training and hosting.

## Containers
Amazon Sagemaker uses docker containers both for training and inference:
- `Dockerfile` is training container, sources from `container_training` directory will be added at training time;
- `Dockerfile.serving` is serving container, `container_serving` directory will added at inference time.
- 'Dockerfile.dronetraining' is a custom training container for custom dataset.

**Note**: by default training container compiles Detectron2 for Volta architecture (Tesla V100 GPUs). If you'd like to run training on other GPU architectures, consider updating this [environment variable](https://github.com/vdabravolski/detectron2-sagemaker/blob/e6252211b819815962207d1a61d5675d213e0f25/Dockerfile#L21). Here is an example on how to compile Detectron2 for all supported architectures:

`ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"`

## Distributed training on COCO2017 dataset
See `d2_byoc_coco2017_training.ipynb` for end-to-end example of how to train your Detectron2 model on Sagemaker. Current implementation supports both multi-node and multi-GPU training on Sagemaker distributed cluster.

### Training cluster config
- To define parameters of your distributed training cluster, use Sagemaker Estimator configuration:
```python
d2 = sagemaker.estimator.Estimator(...
                                   train_instance_count=2, 
                                   train_instance_type='ml.p3.16xlarge',
                                   train_volume_size=100,
                                   ...
                                   )
```
###  Detecrton2 config
Detectron2 config is defined in Sagemaker Hyperparameters dict:
```python
hyperparameters = {"config-file":"COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml", 
                   #"local-config-file" : "config.yaml", # if you'd like to supply custom config file, please add it in container_training folder, and provide file name here
                   "resume":"True", # whether to re-use weights from pre-trained model
                   "eval-only":"False", # whether to perform only D2 model evaluation
                  # opts are D2 model configuration as defined here: https://detectron2.readthedocs.io/modules/config.html#config-references
                  # this is a way to override individual parameters in D2 configuration from Sagemaker API
                   "opts": "SOLVER.MAX_ITER 20000"
                   }
```
There are 3 ways how you can fine-tune your Detectron2 configuration:
- you can use one of Detectron2 authored config files (e.g. `"config-file":"COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"`).
- you can define your own config file and stored it `container_training` folder. In this case you need to define `local-config-file` parameter with name of desired config file. **Note**, that you can choose either `config-file` or `local-config-file`.
- you can modify individual parameters of Detectron2 configuration via `opts` list (e.g. `"opts": "SOLVER.MAX_ITER 20000"` above.


## Serving trained D2 model for inference
See `d2_byoc_coco2017_inference.ipynb` notebook with example how to host D2 pre-trained model on Sagemaker Inference endpoint.

## Training and serving Detectron2 model for custom problem
See `d2_custom_drone_dataset.ipynb` notebook for details.

## Future work
- [ ] try to export Detectron2 models to Torchscript (not all model architectures are supported today). If succesfful, torchscript models can use Sagemaker Elastic Inference hosting endpoints (fractional GPUs). See `export.md` for current status.
- [ ] process video stream using Detecrton2 model hosted on Sagemaker inference endpoint.






