import numpy as np
from PIL import Image


def get_label_names(onehots, class_names):
    """get all the names when a label is a one-hot."""
    label_idx = np.where(onehots)[0]
    label_names = [class_names[idx] for idx in label_idx]
    return label_names


def save_image_tif(ds, idx):
    img = ds[idx]['data'].numpy()
    print(img.max(), img.min())
    labels = ds[idx]['labels']
    for band in range(len(img)):
        # normalize based on min/max pixel values to clamp ranges in [0, 1]
        img[band, ...] = (img[band, ...] - np.min(img[band, ...])) / np.max(img[band, ...])

    img = np.moveaxis(img, 0, 2)  # C, W, H > W, H, C
    img = img[:, :, ::-1]  # BGR to RGB
    img *= 255
    img = img.astype(np.uint8)
    # label_names = get_label_names(labels, ds.class_names)
    # plt.title('\n'.join(label_names))
    # plt.imshow(img)
    # ax = plt.gca()
    # ax.set_axis_off()
    # plt.show()
    im = Image.fromarray(img)
    im.save('images/test.tif')




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
