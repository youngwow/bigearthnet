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


onnx_model_path = r'my_models/model_type_forest.onnx'
x_image = cv.imread("images/field.tif")
x_image = cv.resize(x_image, (120, 120))
x_image = cv.cvtColor(x_image, cv.COLOR_BGR2RGB)
x_numpy = np.array(x_image)
x_numpy = x_numpy[:, :, ::-1]  # RGB to BGR
x_numpy = np.moveaxis(x_numpy, 2, 0)  # W, H, C >  C, W, H

x_numpy = np.expand_dims(x_numpy, 0)


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

onnx_model = onnx.load(onnx_model_path)

onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: x_normalized_onnx}
ort_outs = ort_session.run(None, ort_inputs)

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
