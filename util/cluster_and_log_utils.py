import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import torch

def set_args_mmf(args, train_loader):
    train_loader_targets = train_loader.dataset.labelled_dataset.targets + train_loader.dataset.unlabelled_dataset.datasets[0].targets + train_loader.dataset.unlabelled_dataset.datasets[1].targets

    labeled_num = len(torch.unique(torch.tensor(train_loader.dataset.labelled_dataset.targets)))
    print(f'Labeled Classes Number: {labeled_num}')
    print(f'Labeled Set Length: {len(train_loader.dataset.labelled_dataset.targets)}')

    print(f'Unlabeled Set Length: {len(train_loader_targets)}')

    train_loader_targets = torch.tensor(train_loader_targets)
    known_train_loader_targets = train_loader_targets[train_loader_targets < args.num_labeled_classes]
    unknown_train_loader_targets = train_loader_targets[train_loader_targets >= args.num_labeled_classes]

    k_cls_idx, ins_num = torch.unique(known_train_loader_targets, return_counts=True)

    val, idx = torch.sort(ins_num, descending=True)

    if len(k_cls_idx) == 5:  # CIFAR-10
        many_thre = val[1]
        few_thre = val[2]
    else:
        many_thre = val[int(1 / 3 * len(k_cls_idx))]
        few_thre = val[int(2 / 3 * len(k_cls_idx))]

    known_many_cls = k_cls_idx[ins_num > many_thre]
    known_few_cls = k_cls_idx[ins_num < few_thre]
    known_median_cls = k_cls_idx[(ins_num <= many_thre) & (ins_num >= few_thre)]
    assert len(k_cls_idx) == (len(known_many_cls) + len(known_few_cls) + len(known_median_cls))

    uk_cls_idx, ins_num = torch.unique(unknown_train_loader_targets, return_counts=True)
    val, idx = torch.sort(ins_num, descending=True)

    if len(k_cls_idx) == 5:  # CIFAR-10
        many_thre = val[1]
        few_thre = val[2]
    else:
        many_thre = val[int(1 / 3 * len(uk_cls_idx))]
        few_thre = val[int(2 / 3 * len(uk_cls_idx))]

    unknown_many_cls = uk_cls_idx[ins_num > many_thre]
    unknown_few_cls = uk_cls_idx[ins_num < few_thre]
    unknown_median_cls = uk_cls_idx[(ins_num <= many_thre) & (ins_num >= few_thre)]

    args.known_many_cls = known_many_cls
    args.known_median_cls = known_median_cls
    args.known_few_cls = known_few_cls
    args.unknown_many_cls = unknown_many_cls
    args.unknown_median_cls = unknown_median_cls
    args.unknown_few_cls = unknown_few_cls
    args.k_cls_idx = k_cls_idx
    print(
        f'Known Many: {len(known_many_cls)}  Known Med: {len(known_median_cls)}  Known Few: {len(known_few_cls)} \n Novel Many: {len(unknown_many_cls)}  Novel Med: {len(unknown_median_cls)}  Novel Few: {len(unknown_few_cls)}')


def calculate_acc(ind_map, w, list_targets):
    acc_list = []
    for targets in list_targets:
        if targets is None:
            acc_list.append(-1.)

            continue
        targets = targets.numpy().tolist()
        acc = 0
        total_instances = 0

        for i in targets:
            acc += w[ind_map[i], i]
            total_instances += sum(w[:, i])

        if total_instances != 0:
            acc /= total_instances
            acc_list.append(acc * 100)
        else:
            acc_list.append(-1.)

    return acc_list


def split_cluster_acc_v2(y_true, y_pred, mask, train_loader, args):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    total_acc *= 100

    if y_true.max() == args.k_cls_idx.max().item():
        args.unknown_many_cls = args.unknown_median_cls = args.unknown_few_cls = None
    acc_list = calculate_acc(ind_map=ind_map, w=w, list_targets=[args.known_many_cls, args.known_median_cls, args.known_few_cls,
                                                      args.unknown_many_cls, args.unknown_median_cls, args.unknown_few_cls])

    old_acc = 0
    total_old_instances = 0
    total_pred_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
        total_pred_old_instances += sum(w[ind_map[i], :])

    if total_old_instances != 0:
        old_acc /= total_old_instances
        old_acc *= 100
    else:
        old_acc = -1.

    new_acc = 0
    total_new_instances = 0
    total_pred_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
        total_pred_new_instances += sum(w[ind_map[i], :])

    if total_new_instances != 0:
        new_acc /= total_new_instances
        new_acc *= 100
    else:
        new_acc = -1.

    return total_acc, old_acc, new_acc, acc_list, ind_map


def split_cluster_acc_v2_balanced(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}

    old_acc = np.zeros(len(old_classes_gt))
    total_old_instances = np.zeros(len(old_classes_gt))
    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(len(new_classes_gt))
    total_new_instances = np.zeros(len(new_classes_gt))
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_acc /= total_old_instances
    new_acc /= total_new_instances
    total_acc, old_acc, new_acc = total_acc.mean(), old_acc.mean(), new_acc.mean()
    return total_acc, old_acc, new_acc


EVAL_FUNCS = {
    'v2': split_cluster_acc_v2,
    'v2b': split_cluster_acc_v2_balanced
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs, save_name, T=None,
                        print_output=True, args=None, train_loader=None):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc, acc_list, ind_map = acc_f(y_true, y_pred, mask, train_loader, args)
        log_name = f'{save_name}_{f_name}'

        if i == 0:
            to_return = (all_acc, old_acc, new_acc, acc_list, ind_map)

        if print_output:
            old_std = np.array([acc_list[0], acc_list[1], acc_list[2]]).std()
            new_std = np.array([acc_list[3], acc_list[4], acc_list[5]]).std()
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.1f} | Old {old_acc:.1f} | New {new_acc:.1f} | '
            print_str2 = f'KMany {acc_list[0]:.1f} | KMed {acc_list[1]:.1f} | KFew {acc_list[2]:.1f} | Std {old_std:.1f}'
            print_str3 = f'UMany {acc_list[3]:.1f} | UMed {acc_list[4]:.1f} | UFew {acc_list[5]:.1f} | Std {new_std:.1f}'

            try:
                args.logger.info(print_str)
                args.logger.info(print_str2)
                args.logger.info(print_str3)
            except:
                print(print_str)
                print(print_str2)
                print(print_str3)
    return to_return
