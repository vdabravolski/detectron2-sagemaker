1. When trying to run scripting of backbone network, face issue with unsupported operation:
sample command: 
`python torchscript_export.py --config ./trained_models/R50-C4/faster_rcnn_R_50_C4_1x.yaml --image ./trained_models/model_R_50_FPN_1x/coco_sample.jpg --task script --weights ./trained_models/R50-C4/R50-C4.pkl` 

Output:
```
{
torch.jit.frontend.UnsupportedNodeError: GeneratorExp aren't supported:
  File "/home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/detectron2/layers/wrappers.py", line 88
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                             ~ <--- HERE
                return empty + _dummy
            else:

}
```


2. When trying to run tracing, face the issue that backbone network returns dict, and not tensor. Only tensor output is supported in Torch 1.5
Sample command:
`python torchscript_export.py --config ./trained_models/R50-C4/faster_rcnn_R_50_C4_1x.yaml --image ./trained_models/model_R_50_FPN_1x/coco_sample.jpg --task trace --weights ./trained_models/R50-C4/R50-C4.pkl`

Output:
```
{
Traceback (most recent call last):
  File "torchscript_export.py", line 88, in <module>
    run_trace(args)
  File "torchscript_export.py", line 66, in run_trace
    traced_bcbn = torch.jit.trace(model.backbone, images_prep.tensor)
  File "/home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/jit/__init__.py", line 875, in trace
    check_tolerance, _force_outplace, _module_class)
  File "/home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/jit/__init__.py", line 1027, in trace_module
    module._c._create_method_from_trace(method_name, func, example_inputs, var_lookup_fn, _force_outplace)
RuntimeError: Only tensors, lists and tuples of tensors can be output from traced functions
}
```

However, recently a new feature was added in torch master to support dict outputs: https://github.com/pytorch/pytorch/issues/27743 Need to recompile torch from source and retest.