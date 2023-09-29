from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.imagenet import get_imagenet_100_datasets
from copy import deepcopy
import pickle
import os


get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
'imagenet100': get_imagenet_100_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform, args=args)

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = None

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):

    if args.dataset_name == 'cifar10':
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)
    elif args.dataset_name == 'cifar100':
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)
    elif args.dataset_name == 'imagenet100':
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)


    return args
