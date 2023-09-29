import torchvision
import numpy as np

import os

from copy import deepcopy
from data.data_utils import subsample_instances
from config import imagenet_root
import torch

class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


def subsample_dataset(dataset, idxs, args):


    img_path_ = []
    for i in idxs:
        img_path_.append(dataset.img_path[i])
    dataset.img_path = img_path_

    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset


def subsample_classes(dataset, include_classes=list(range(1000)), args=None):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs, args)

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_lt_dist(cls_num, img_max, imb_factor):
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return torch.tensor(img_num_per_cls)

def get_imagenet_100_datasets(train_transform, test_transform, train_classes=range(50),
                           prop_train_labels=0.5, split_train_val=False, seed=0, args=None):

    from data.LT_dataset import LT_Dataset
    txt_path = "splits/imagenet100/imageNet_100_LT_train.txt"
    test_txt_path = "splits/imagenet100/ImageNet_100_test.txt"
    total_class = 100
    args.anno_ratio = 50
    seed = 416
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    whole_training_set = LT_Dataset(root=imagenet_root, txt=txt_path, transform=train_transform)

    whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    known_classes = train_classes
    unknown_classes = range(len(train_classes), total_class)
    whole_known_classes_set = subsample_classes(deepcopy(whole_training_set), include_classes=known_classes,
                                                args=args)

    labeled_known_idxs = []
    unlabeled_known_idxs = []
    known_classes = np.unique(whole_known_classes_set.targets)
    for cls in known_classes:
        cls_idxs = np.where(whole_known_classes_set.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False,
                              size=((int((1 - args.anno_ratio / 100) * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        labeled_known_idxs.extend(t_)
        unlabeled_known_idxs.extend(v_)

    lt_labeled_known_dataset = subsample_dataset(deepcopy(whole_known_classes_set), labeled_known_idxs, args)
    lt_unlabeled_known_dataset = subsample_dataset(deepcopy(whole_known_classes_set), unlabeled_known_idxs,
                                                   args)

    lt_unlabeled_unknown_dataset = subsample_classes(deepcopy(whole_training_set),
                                                     include_classes=range(len(train_classes), total_class),
                                                     args=args)

    # Get test set for all classes
    test_dataset = LT_Dataset(root=imagenet_root, txt=test_txt_path, transform=test_transform)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = lt_labeled_known_dataset
    train_dataset_unlabelled = torch.utils.data.ConcatDataset(
        [lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])
    val_dataset_labelled = None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'bl_train_unlabelled': None
    }
    return all_datasets