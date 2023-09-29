import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('./')
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model.loss import info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from copy import deepcopy
from sklearn.cluster import KMeans
# from cuml.cluster import KMeans as cuKMeans
from data.cifar import CustomCIFAR100, cifar_100_root
import random
from model.dist_est import dist_est
from model.reg_loss import compute_reg_loss
from model.softconloss import compute_softconloss
import os
import sys

class CE_Head(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return logits


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()


def train_dual(ce_backbone, ce_head, cl_backbone, cl_head, train_loader, test_loader, args):
    def set_model(train=False, eval=False):
        assert (train ^ eval)
        if train:
            student_ce.train()
            student_cl.train()
        elif eval:
            student_ce.eval()
            student_cl.eval()

    from util.cluster_and_log_utils import set_args_mmf
    set_args_mmf(args, train_loader)

    save_path = 'checkpoints/' + os.path.join(*args.model_path.split('/')[3:-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    student_ce = nn.Sequential(ce_backbone, ce_head).to(device)
    student_cl = nn.Sequential(cl_backbone, cl_head).to(device)

    params_groups_cl = list(cl_head.parameters()) + list(cl_backbone.parameters())
    params_groups_ce = get_params_groups(student_ce)
    optimizer_ce = SGD(params_groups_ce, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler_ce = lr_scheduler.CosineAnnealingLR(
        optimizer_ce,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    optimizer_cl = SGD(params_groups_cl, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler_cl = lr_scheduler.CosineAnnealingLR(
        optimizer_cl,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )
    args.current_epoch = 0
    best_test_acc_all_cl = -1
    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args=args,
    )

    set_model(eval=True)
    est_count = dist_est(ce_backbone, cl_backbone, ce_head, train_all_test_trans_loader, args)
    set_model(train=True)


    for epoch in range(args.epochs):
        args.current_epoch = epoch
        loss_record_ce = AverageMeter()
        loss_record_cl = AverageMeter()
        for batch_idx, batch in enumerate(train_loader):
            images_, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images_, dim=0).cuda(non_blocking=True)

            x = ce_backbone.prepare_tokens(images)

            for i, blk in enumerate(ce_backbone.blocks):
                if i < args.grad_from_block:
                    x = blk(x)   # get fixed feature

            for i, blk in enumerate(ce_backbone.blocks):
                if i >= args.grad_from_block:
                    ce_backbone_feature = blk(x)
            ce_backbone_feature = ce_backbone.norm(ce_backbone_feature)
            ce_backbone_feature = ce_backbone_feature[:, 0]
            student_out = ce_head(ce_backbone_feature)

            for i, blk in enumerate(cl_backbone.blocks):
                if i >= args.grad_from_block:
                    cl_backbone_feature = blk(x)
            cl_backbone_feature = cl_backbone.norm(cl_backbone_feature)
            cl_backbone_feature = cl_backbone_feature[:, 0]
            cl_proj_feature = cl_head(cl_backbone_feature)

            ####################### COMPUTE LOSS #######################
            pstr = ''
            teacher_out = student_out.detach()

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)

            # clustering, sup
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)

            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            me_max_loss = compute_reg_loss(student_out, est_count, args)

            pstr += f'cls_loss: {cls_loss.item():.2f} '
            pstr += f'cps_loss: {cluster_loss.item():.2f} '
            pstr += f'reg_loss: {me_max_loss.item():.2f} '
            cluster_loss += args.memax_weight * me_max_loss
            loss_ce = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss

            loss_record_ce.update(loss_ce.item(), class_labels.size(0))
            optimizer_ce.zero_grad()
            loss_ce.backward()
            optimizer_ce.step()

            loss_cl = 0
            # for CL part
            # represent learning, unsup
            cl_proj_feature = torch.nn.functional.normalize(cl_proj_feature, dim=-1)

            contrastive_logits, contrastive_labels = info_nce_logits(features=cl_proj_feature)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup
            sup_cl_proj_feature = torch.cat([f[mask_lab].unsqueeze(1) for f in cl_proj_feature.chunk(2)], dim=1)
            sup_con_labels = class_labels[mask_lab]

            if epoch >= args.ce_warmup:
                sup_con_loss = SupConLoss()(sup_cl_proj_feature, labels=sup_con_labels)
                soft_con_loss = compute_softconloss(student_out, cl_proj_feature, sup_con_labels, sup_cl_proj_feature, mask_lab, args)
                loss_cl += ((1 - args.sup_weight) * contrastive_loss + (args.sup_weight / 2) * sup_con_loss)
                loss_cl += (args.sup_weight / 2) * soft_con_loss
            else:
                sup_con_loss = SupConLoss()(sup_cl_proj_feature, labels=sup_con_labels)
                loss_cl += ((1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss)

            pstr += f'sup_con_loss: {sup_con_loss.item():.2f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.2f} '

            loss_record_cl.update(loss_cl.item(), class_labels.size(0))
            optimizer_cl.zero_grad()
            loss_cl.backward()
            optimizer_cl.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss_ce {:.5f} loss_cl {:.5f}\t {}'
                                 .format(epoch, batch_idx, len(train_loader), loss_ce.item(), loss_cl.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss_ce: {:.2f} Avg Loss_cl: {:.2f}'.format(epoch, loss_record_ce.avg, loss_record_cl.avg))

        if (epoch+1) % args.est_freq == 0:
            set_model(eval=True)
            est_count = dist_est(ce_backbone, cl_backbone, ce_head, train_all_test_trans_loader, args)

        if epoch % args.test_freq == 0:
            args.logger.info('Testing on disjoint test set...')
            with torch.no_grad():
                all_acc_test_cl, old_acc_test_cl, new_acc_test_cl, acc_list_cl, cl_ind_map = test(
                    student_cl,
                    test_loader,
                    epoch=epoch,
                    save_name='Test ACC',
                    args=args,
                    train_loader=train_loader)
            
            args.logger.info(
                    'Test Accuracies CL: All {:.1f} | Old {:.1f} | New {:.1f}'.format(all_acc_test_cl,
                                                                                      old_acc_test_cl,
                                                                                      new_acc_test_cl))

        # Step schedule
        exp_lr_scheduler_ce.step()
        exp_lr_scheduler_cl.step()

        if epoch % args.test_freq == 0 and all_acc_test_cl > best_test_acc_all_cl:

            best_test_acc_new_cl = new_acc_test_cl
            best_test_acc_old_cl = old_acc_test_cl
            best_test_acc_all_cl = all_acc_test_cl

            save_dict_cl = {
                'ce_backbone': ce_backbone.state_dict(),
                'ce_head': ce_head.state_dict(),
                'cl_backbone': cl_backbone.state_dict(),
                'cl_head': cl_head.state_dict(),
            }

            torch.save(save_dict_cl, save_path + f'/model_epoch{epoch}.pt')
            args.logger.info("model saved to {}.".format(save_path))

        if epoch >= args.stop_epoch:
            break

    args.logger.info(
        f'Metrics with best model on test set: All: {best_test_acc_all_cl:.1f} Old: {best_test_acc_old_cl:.1f} New: {best_test_acc_new_cl:.1f} ')


def test(model, test_loader, epoch, save_name, args, train_loader):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    print('Collating features...')
    # First extract all features
    for batch_idx, batch in enumerate(test_loader):
        batch = batch[:3]
        (images, label, _) = batch
        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model[0](images)  # follow GCD: clustering on normalized backbone feature

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.detach().cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    all_acc, old_acc, new_acc, acc_list, ind_map = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args, train_loader=train_loader)

    return all_acc, old_acc, new_acc, acc_list, ind_map


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', type=list, default=['v2'])
    parser.add_argument('--dataset_name', type=str, default='cifar100')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--memax_weight', type=float, default=1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='cifar100', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--p', default=1.1, type=float)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--ce_warmup', default=1, type=int)
    parser.add_argument('--est_freq', default=10, type=int)
    parser.add_argument('--labeled_classes', default=80, type=int)
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--tro', default=0.5, type=float)
    parser.add_argument('--stop_epoch', default=200, type=int)
    parser.add_argument('--imb_ratio', default=100, type=int)

    # ----------------------
    # INIT
    # ----------------------

    args = parser.parse_args()
    pid = os.getpid()
    print('MY PID：', pid)

    if args.config_file != '':
        with open('configs/' + args.config_file, 'r') as f:
            args = parser.parse_args(f.read().split())


    device = torch.device('cuda:0')
    args = get_class_splits(args)
    total_class = 100 if args.dataset_name != 'cifar10' else 10

    args.train_classes = range(args.labeled_classes)
    args.unlabeled_classes = range(args.labeled_classes, total_class)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    init_experiment(args, runner_name=['BaCon'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    seed = torch.randint(0, 100000, (1,)).item()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)

    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_all_test_trans = deepcopy(train_dataset)
    train_labelled_test_trans = deepcopy(train_dataset.labelled_dataset)
    train_unlabelled_test_trans = deepcopy(train_dataset.unlabelled_dataset)

    train_all_test_trans.labelled_dataset.transform = test_transform

    train_all_test_trans.unlabelled_dataset.datasets[0].transform = test_transform
    train_all_test_trans.unlabelled_dataset.datasets[1].transform = test_transform
    train_unlabelled_test_trans.datasets[0].transform = test_transform
    train_unlabelled_test_trans.datasets[1].transform = test_transform
    train_labelled_test_trans.transform = test_transform


    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    train_all_test_trans_loader = DataLoader(train_all_test_trans, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)
    train_labelled_test_trans_loader = DataLoader(train_labelled_test_trans, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)
    train_unlabelled_test_trans_loader = DataLoader(train_unlabelled_test_trans, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------

    from model import vision_transformer as vits


    cl_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim, out_dim=65536, nlayers=args.num_mlp_layers)
    ce_head = CE_Head(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

    cl_backbone = deepcopy(backbone)
    ce_backbone = deepcopy(backbone)

    # ----------------------
    # TRAIN
    # ----------------------

    ce_backbone = ce_backbone.to(device)
    ce_head = ce_head.to(device)
    cl_backbone = cl_backbone.to(device)
    cl_head = cl_head.to(device)
    
    train_dual(ce_backbone, ce_head, cl_backbone, cl_head, train_loader, test_loader_labelled, args)


