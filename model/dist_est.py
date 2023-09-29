import torch
import numpy as np
from sklearn.cluster import KMeans
from logits_adj_utils import compute_est_adjustment
# from cuml.cluster import KMeans as cuKMeans
import argparse

device = torch.device('cuda:0')


@torch.no_grad()
def dist_est(ce_backbone, cl_backbone, ce_head, train_loader_test_trans, args, debug=False):

    est_count = torch.ones(args.num_classes).cuda()
    with torch.no_grad():
        all_feats_cl = []
        targets = np.array([])
        mask_lab = np.array([])
        uq_idxs = np.array([])
        # First extract all features
        for batch_idx, batch in enumerate(train_loader_test_trans):
            images, class_labels, uq_idxs_, mask_lab_ = batch
            mask_lab_ = mask_lab_[:, 0]
            images = images.cuda(non_blocking=True)
            class_labels, mask_lab_ = class_labels.cuda(non_blocking=True), mask_lab_.cuda(
                non_blocking=True).bool()
            cl_backbone_feature = cl_backbone(images)

            uq_idxs = np.append(uq_idxs, uq_idxs_.cpu().numpy())
            targets = np.append(targets, class_labels.cpu().numpy())
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

            cl_feats = torch.nn.functional.normalize(cl_backbone_feature, dim=-1)
            all_feats_cl.append(cl_feats.detach().cpu().numpy())

        mask_lab = mask_lab.astype(bool)
        all_feats = np.concatenate(all_feats_cl)

        kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes,
                        random_state=0).fit(all_feats)
        all_preds = kmeans.labels_

        all_preds = torch.from_numpy(all_preds)
        _, ins_num = torch.unique(all_preds, return_counts=True)


    def align(y_pred, y_true):
        assert y_pred.size == y_true.size
        # D = max(y_pred.max(), y_true.max()) + 1
        D = args.num_classes
        w = np.zeros((D, D), dtype=float)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        w /= ins_num[:D].view(-1, 1).repeat(1, D).numpy()

        from scipy.optimize import linear_sum_assignment as linear_assignment
        ind = linear_assignment(w.max() - w)
        y_pred_id, y_true_id = ind
        return torch.from_numpy(y_pred_id), torch.from_numpy(y_true_id)

    labeled_preds = all_preds[mask_lab]
    labeled_targets = targets[mask_lab]

    y_pred_id, y_true_id = align(y_pred=labeled_preds.cpu().numpy().astype(int),
                                                    y_true=labeled_targets.astype(int))
    _, idx = y_true_id.sort()
    assert torch.all(idx == y_pred_id[idx])
    # _, idx = labeled_targets.sort()[1]
    labeled_targets = torch.from_numpy(labeled_targets)
    _, known_labeled_ins_num = torch.unique(labeled_targets, return_counts=True)

    idx1 = torch.argsort(known_labeled_ins_num)
    known_cluster_ins_num = ins_num[idx[:args.num_labeled_classes]]
    known_cluster_ins_num = known_cluster_ins_num.to(est_count.dtype).cuda()
    idx2 = torch.argsort(known_cluster_ins_num)

    est_count[:args.num_labeled_classes][idx1] = known_cluster_ins_num[idx2]
    est_count[args.num_labeled_classes:] = ins_num[idx[args.num_labeled_classes:]].sort(descending=True)[0]

    args.est_dist = est_count.detach().cuda()
    args.est_adjustment = compute_est_adjustment(tro=args.tro, args=args)
    return est_count.detach().cuda()
