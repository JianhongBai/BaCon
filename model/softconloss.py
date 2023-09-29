import torch
from model.loss import SemiConLoss
sample_count = torch.zeros(100).cuda()


def compute_softconloss(student_out, cl_proj_feature, sup_con_labels, sup_cl_proj_feature, mask_lab, args):

    logits = (student_out / 0.1) - args.est_adjustment
    view1_logits, view2_logits = logits.softmax(dim=1).chunk(2)
    soft_labels = (view1_logits + view2_logits) / 2
    
    known_class_sampling_rate = ((1 / args.est_dist) * args.est_dist.min()) ** (args.alpha)
    existing_class_idx, _ = torch.unique(sup_con_labels, return_counts=True)
    sampling_rate = ((1 / args.est_dist) * args.est_dist.min()) ** (args.beta)
    sampling_rate[existing_class_idx] = known_class_sampling_rate[existing_class_idx]
    batch_confidence, batch_preds = soft_labels[~mask_lab].max(dim=1)

    sampling_mask = torch.zeros(len(soft_labels)).bool().cuda()
    mask = torch.arange(len(soft_labels)).cuda()

    for cls_idx in torch.unique(batch_preds):
        cls_ins_num = (batch_preds == cls_idx).sum()
        cls_sample_num = torch.bernoulli(torch.tensor([sampling_rate[cls_idx]] * cls_ins_num)).sum().int()
        cls_conf = batch_confidence[batch_preds == cls_idx]
        val, idx = torch.topk(cls_conf, k=cls_sample_num)
        sampling_mask[mask[~mask_lab][batch_preds == cls_idx][idx]] = True

    semicon_soft_labels = soft_labels[(~mask_lab) & sampling_mask]
    semicon_feats = torch.cat([f[(~mask_lab) & sampling_mask].unsqueeze(1) for f in cl_proj_feature.chunk(2)], dim=1)

    sup_labels_one_hot = soft_labels[mask_lab]
    sup_cl_proj_feature = torch.cat([sup_cl_proj_feature, semicon_feats], dim=0)
    sup_con_labels = torch.cat([sup_labels_one_hot, semicon_soft_labels], dim=0)
    soft_mask = (sup_con_labels.unsqueeze(1) * sup_con_labels.unsqueeze(0)).sum(dim=2)
    soft_mask[torch.eye(len(soft_mask), device=soft_mask.device).bool()] = 1

    sup_con_loss = SemiConLoss(args=args)(sup_cl_proj_feature, soft_mask=soft_mask.detach())
    return sup_con_loss
