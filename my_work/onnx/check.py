import onnx
from bigearthnet.models.bigearthnet_module import BigEarthNetModule
from bigearthnet.datamodules.bigearthnet_datamodule import BigEarthNetDataModule
from hydra.utils import instantiate
import torch
import torch.nn as nn
from collections.abc import Sequence
import onnx
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import json


def calc_same_padding(kernel_size, stride, input_size):
    if isinstance(kernel_size, Sequence):
        kernel_size = kernel_size[0]

    if isinstance(stride, Sequence):
        stride = stride[0]

    if isinstance(input_size, Sequence):
        input_size = input_size[0]

    pad = ((stride - 1) * input_size - stride + kernel_size) / 2
    return int(pad)


def replace_conv2d_with_same_padding(m: nn.Module, input_size=512):
    if isinstance(m, nn.Conv2d):
        if m.padding == "same":
            m.padding = calc_same_padding(
                kernel_size=m.kernel_size,
                stride=m.stride,
                input_size=input_size
            )


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


ckpt_path = r'my_models/best-model.ckpt'
onnx_model_path = r'my_models/model_type_forest.onnx'
dataset_dir = r'../../datasets/'  # root directory where to download the datasets
dataset_name = "bigearthnet-mini"
# One of bigearthnet-mini, bigearthnet-medium, bigearthnet-full
batch_size = 16

num_workers = 0
# Load the model from the checkpoint
model = BigEarthNetModule.load_from_checkpoint(ckpt_path)

model.eval()
# fetch the transforms used in the model

transforms = instantiate(model.cfg.transforms.obj)

print(type(transforms))
# Input to the model
x_random = torch.randn(3, 120, 120, requires_grad=True)
x_image = cv.imread("images/field.tif")
x_image = cv.resize(x_image, (120, 120))
x_image = cv.cvtColor(x_image, cv.COLOR_BGR2RGB)
x_numpy = np.array(x_image)
x_numpy = x_numpy[:, :, ::-1]  # RGB to BGR
x_numpy = np.moveaxis(x_numpy, 2, 0)  # W, H, C >  C, W, H
print(list(x_numpy.shape))
assert list(x_random.shape) == list(x_numpy.shape)

x_tensor = torch.tensor(x_numpy.copy(), dtype=torch.float32, requires_grad=True)

x_tensor = torch.unsqueeze(x_tensor, dim=0)
x_numpy = np.expand_dims(x_numpy, 0)
assert list(x_numpy.shape) == list(x_tensor.shape)
print("Before transform for pytorch")
# print(x)
x_torch = transforms(x_tensor)
print("after transform for pytorch")
print(x_torch)
print(x_torch.dtype)
torch_out = model.model(x_torch)
# Export the model
model.model.apply(lambda m: replace_conv2d_with_same_padding(m, 120))

torch.onnx.export(model.model,  # model being run
                  x_torch,  # model input (or a tuple for multiple inputs)
                  onnx_model_path,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})
onnx_model = onnx.load(onnx_model_path)

onnx.checker.check_model(onnx_model)


ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])


with open(r"mean_stddev.json", "r") as f:
    mean_stddev = json.load(f)

with open(r"class_list.json", "r") as f:
    class_names = json.load(f)

mean = np.array(mean_stddev["mean"]).reshape(-1, 1, 1)
std = np.array(mean_stddev["std"]).reshape(-1, 1, 1)
# print(mean_stddev)
print("Before transform for onnx")
x_normalized_onnx = (x_numpy - mean) / std
x_normalized_onnx = x_normalized_onnx.astype(np.float32)
print("after transform for onnx")
# print(x_normalized_onnx)

# compare Manual normalized and PyTorch normalized
np.testing.assert_allclose(to_numpy(x_torch), x_normalized_onnx, rtol=1e-03, atol=1e-05)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: x_normalized_onnx}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

logits_onnx = ort_outs[0]

print("Predictions by onnx model:\n")
sigmoid = lambda x: 1 / (1 + np.exp(-x))
preds = np.squeeze(sigmoid(logits_onnx) > 0.5)
print(preds)
indices = [i for i, x in enumerate(preds) if x]
print(indices)
# indices = torch.where(preds is True)[1]  # get only positive predictions
for idx in indices:  # iterate through the prediction indices
    print(class_names[idx])


print("Predictions by pytorch model:\n")
sigmoid = lambda x: 1 / (1 + np.exp(-x))
preds = np.squeeze(sigmoid(to_numpy(torch_out)) > 0.5)
print(preds)
indices = [i for i, x in enumerate(preds) if x]
print(indices)
# indices = torch.where(preds is True)[1]  # get only positive predictions
for idx in indices:  # iterate through the prediction indices
    print(class_names[idx])
