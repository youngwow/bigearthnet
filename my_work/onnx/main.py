import onnx
from bigearthnet.models.bigearthnet_module import BigEarthNetModule
from bigearthnet.datamodules.bigearthnet_datamodule import BigEarthNetDataModule
from hydra.utils import instantiate
import torch
import torch.nn as nn
from collections.abc import Sequence
import onnx
import onnxruntime
import matplotlib.pyplot as plt
import cv2 as cv
import json
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio import transform
# from osgeo import gdal
import numpy as np
from PIL import Image


def get_label_names(onehots, class_names):
    """get all the names when a label is a one-hot."""
    label_idx = np.where(onehots)[0]
    label_names = [class_names[idx] for idx in label_idx]
    return label_names


def save_image_tif(ds, idx):
    # sourceimg = Image.open("images/source.tif")
    # # Get the TIFF tags from the source image
    # tiffinfo = sourceimg.tag_v2
    # geoinfo = tiffinfo[34737]
    # print(tiffinfo)
    # print("-"*20)
    # print(geoinfo)
    # step1 = gdal.Open('images/get_geo.tif', gdal.GA_ReadOnly)
    # GT_input = step1.GetGeoTransform()
    img = ds[idx]['data'].numpy()
    labels = ds[idx]['labels']
    for index, label in enumerate(labels):  # iterate through the prediction indices\
        if label:
            print(ds.class_names[index])
    print(labels)
    for band in range(len(img)):
        # normalize based on min/max pixel values to clamp ranges in [0, 1]
        img[band, ...] = (img[band, ...] - np.min(img[band, ...])) / np.max(img[band, ...])

    img = np.moveaxis(img, 0, 2)  # C, W, H > W, H, C
    img = img[:, :, ::-1]  # BGR to RGB
    img *= 255
    img = img.astype(np.uint8)
    print(img.shape)

    # dst_crs = 'EPSG:32722'

    # with rasterio.open("images/source.tif", mode="r", nodata=0) as src:
    #     profile = src.profile

    # with rasterio.open(
    #         'output_map.tif',
    #         'w',
    #         driver='GTiff',
    #         height=img.shape[1],
    #         width=img.shape[2],
    #         count=3,
    #         dtype=np.uint8,
    #         crs=dst_crs
    # ) as dest_file, rasterio.open("images/source.tif", mode="r", nodata=0) as src:
    #     dest_file.profile = src.profile
    #     dest_file.write(img)
    # dest_file.close()
    im = Image.fromarray(img)
    im.save('images/test_info.tif')
    im = Image.open('images/test_info.tif')
    tiffinfo = im.tag_v2
    tiffinfo.tag_v2[34737] = """PROJCS["WGS 84 / UTM zone 29N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32629"]]"""
    im.save('images/test_info.tif', tiffinfo=tiffinfo)
    #
    # targetimg = Image.open('images/test_info.tiff')
    # tiffinfo_target = targetimg.tag_v2
    # tiffinfo_target[34737] = geoinfo
    # print(tiffinfo_target)
    # targetimg.save("target.tif", tiffinfo=tiffinfo_target)


dataset_dir = r'../../datasets/'  # root directory where to download the datasets
dataset_name = "bigearthnet-mini"
# One of bigearthnet-mini, bigearthnet-medium, bigearthnet-full
batch_size = 16
num_workers = 0

datamodule_no_tfm = BigEarthNetDataModule(
    dataset_dir, dataset_name, batch_size, num_workers, transforms=None
)
datamodule_no_tfm.setup()

ds_no_tfm = datamodule_no_tfm.test_dataloader().dataset

# Save image
sample_index = 0
save_image_tif(ds_no_tfm, idx=sample_index)


# logits = ort_outs[0]
# # fetch the transforms used in the model
# transforms = instantiate(model.cfg.transforms.obj)
#
# print(transforms)
#
# datamodule = BigEarthNetDataModule(
#     dataset_dir, dataset_name, batch_size, num_workers, transforms
# )
# datamodule.setup()
# ds = datamodule.test_dataloader().dataset
#
# # Load without transforms so we can see our images too
# datamodule_no_tfm = BigEarthNetDataModule(
#     dataset_dir, dataset_name, batch_size, num_workers, transforms=None
# )
# datamodule_no_tfm.setup()
# ds_no_tfm = datamodule_no_tfm.test_dataloader().dataset
#
# # Displays image and ground truth
# sample_index = 0
#
# save_image_tif(ds_no_tfm, idx=sample_index)
#
# print("Predictions:\n")
# sigmoid = lambda x: 1 / (1 + np.exp(-x))
# preds = np.squeeze(sigmoid(logits) > 0.5)
# print(preds)
# indices = [i for i, x in enumerate(preds) if x]
# print(indices)
# # indices = torch.where(preds is True)[1]  # get only positive predictions
# for idx in indices:  # iterate through the prediction indices
#     print(ds.class_names[idx])
