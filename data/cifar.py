from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np
import torch
from data.data_utils import subsample_instances
from config import cifar_100_root, cifar_10_root

class CustomCIFAR10(CIFAR10):

    def __init__(self, sublist=[], *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.txt = sublist
        self.uq_idxs = np.array(range(len(self)))
        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()
            self.uq_idxs = self.uq_idxs[sublist]

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, sublist=[], return_false_mask=False, return_true_mask=False, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)
        self.txt = sublist

        self.uq_idxs = np.array(range(len(self)))
        self.return_false_mask = return_false_mask
        self.return_true_mask = return_true_mask

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()
            self.uq_idxs = self.uq_idxs[sublist]

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

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


def get_cifar_100_datasets(train_transform, test_transform, args=None):

    whole_training_set = CustomCIFAR100(root=cifar_100_root, train=True, transform=train_transform, download=True)
    whole_training_set.target_transform = None

    l_k_uq_idxs = torch.load(f'data_uq_idxs/cifar100_k80_imb{args.imb_ratio}/l_k_uq_idxs.pt')
    unl_unk_uq_idxs = torch.load(f'data_uq_idxs/cifar100_k80_imb{args.imb_ratio}/unl_unk_uq_idxs.pt')
    unl_k_uq_idxs = torch.load(f'data_uq_idxs/cifar100_k80_imb{args.imb_ratio}/unl_k_uq_idxs.pt')

    train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k_uq_idxs)
    lt_unlabeled_known_dataset = subsample_dataset(deepcopy(whole_training_set), unl_k_uq_idxs)
    lt_unlabeled_unknown_dataset = subsample_dataset(deepcopy(whole_training_set), unl_unk_uq_idxs)
    train_dataset_unlabelled = torch.utils.data.ConcatDataset([lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])

    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'test': test_dataset,
    }

    return all_datasets

def get_cifar_10_datasets(train_transform, test_transform, args=None):

    whole_training_set = CustomCIFAR10(root=cifar_10_root, train=True, transform=train_transform, download=True)
    whole_training_set.target_transform = None

    l_k_uq_idxs = torch.load(f'data_uq_idxs/cifar10_k5_imb{args.imb_ratio}/l_k_uq_idxs.pt')
    unl_unk_uq_idxs = torch.load(f'data_uq_idxs/cifar10_k5_imb{args.imb_ratio}/unl_unk_uq_idxs.pt')
    unl_k_uq_idxs = torch.load(f'data_uq_idxs/cifar10_k5_imb{args.imb_ratio}/unl_k_uq_idxs.pt')

    train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k_uq_idxs)
    lt_unlabeled_known_dataset = subsample_dataset(deepcopy(whole_training_set), unl_k_uq_idxs)
    lt_unlabeled_unknown_dataset = subsample_dataset(deepcopy(whole_training_set), unl_unk_uq_idxs)
    train_dataset_unlabelled = torch.utils.data.ConcatDataset([lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])
    
    test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'test': test_dataset,
    }

    return all_datasets
