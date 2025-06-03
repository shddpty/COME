import os
import os.path as osp
import MLdataset
import argparse
import time
from model import Model
import utils
from utils import AverageMeter
import evaluation
import torch
import numpy as np
from loss import Loss
from torch import nn
from torch.optim import AdamW
import copy
from MLdataset import RecoverDataset
from torch.utils.data import DataLoader


def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0.0)


def calculate_loss(loss_fn, pred, data, rec_v, embs, label, inc_V_ind, inc_L_ind, beta):
    views = len(data)
    mse_loss = loss_fn.weighted_mse_loss(inc_V_ind, data, rec_v)
    contra_loss = 0
    for i in range(views):
        for j in range(views):
            if i == j:
                continue
            contra_loss += loss_fn.weighted_contrastive_loss(embs[:, i, :], embs[:, j, :], inc_V_ind[:, i],
                                                             inc_V_ind[:, j])
    bce_loss = loss_fn.weighted_bce_loss(label, pred, inc_L_ind)
    loss = beta * contra_loss + bce_loss + mse_loss
    return loss


def train_first(loader, model_first, model_second, loss_first, loss_second, optimizer_first, optimizer_second,
                scheduler, epoch, logger, beta):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model_first.train()
    model_second.train()
    end = time.time()

    threshold = max(0.2, 0.5 - epoch * 0.005)
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        inc_V_ind = inc_V_ind.float().to(device)
        inc_L_ind = inc_L_ind.float().to(device)
        label = label.to(device)
        data = [v_data.to(device) for v_data in data]

        pred_t1, rec_v_t1, embs_t1 = model_first(data, inc_V_ind)
        pseudo_label = pred_t1.detach()
        pseudo_label = (pseudo_label > threshold).float() * pseudo_label * inc_L_ind.logical_not() + label * inc_L_ind
        new_inc_L_ind = inc_L_ind.logical_or(pseudo_label)
        pred_s, rec_s, embs_s = model_second(data, inc_V_ind)
        s_loss_u = calculate_loss(loss_second, pred_s, data, rec_s, embs_s, pseudo_label, inc_V_ind, new_inc_L_ind,
                                  beta)
        optimizer_second.zero_grad()
        s_loss_u.backward()
        optimizer_second.step()

        pseudo_label = pred_s.detach()
        pseudo_label = (pseudo_label > threshold).float() * pseudo_label * inc_L_ind.logical_not() + label * inc_L_ind
        new_inc_L_ind = inc_L_ind.logical_or(pseudo_label)
        t_loss = calculate_loss(loss_first, pred_t1, data, rec_v_t1, embs_t1, pseudo_label, inc_V_ind, new_inc_L_ind,
                                beta)
        optimizer_first.zero_grad()
        t_loss.backward()
        optimizer_first.step()

        loss = s_loss_u
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
    logger.info('Epoch:[{0}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Data {data_time.avg:.3f}\t'
                'Loss {losses.avg:.3f}\t'.format(
        epoch, batch_time=batch_time, data_time=data_time, losses=losses))
    return losses, model_first, model_second


def _test_first(loader, model_first, model_second, epoch, logger, mode='train'):
    batch_time = AverageMeter()
    total_labels = []
    total_preds = []

    model_first.eval()
    model_second.eval()
    end = time.time()

    with torch.no_grad():
        for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
            # data_time.update(time.time() - end)
            inc_V_ind = inc_V_ind.float().to(device)
            data = [v_data.to(device) for v_data in data]
            pred, _, _ = model_second(data, inc_V_ind)
            pred = pred.cpu()

            total_labels = np.concatenate((total_labels, label.numpy()), axis=0) if len(
                total_labels) > 0 else label.numpy()
            total_preds = np.concatenate((total_preds, pred.detach().numpy()), axis=0) if len(total_preds) > 0 else \
                pred.detach().numpy()
            batch_time.update(time.time() - end)
            end = time.time()
    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)
    if mode == 'train':
        evaluation_results = [evaluation.compute_average_precision(total_preds, total_labels)]
        logger.info('Epoch:[{0}]\t'
                    'Mode:{mode}\t'
                    'Time {batch_time.avg:.3f}\t'
                    'AP {ap:.4f}\t'.format(
            epoch, mode=mode, batch_time=batch_time,
            ap=evaluation_results[0],
        ))
    else:
        evaluation_results = evaluation.do_metric(total_preds, total_labels)  # compute auc is very slow
        logger.info('Epoch:[{0}]\t'
                    'Mode:{mode}\t'
                    'Time {batch_time.avg:.3f}\t'
                    'AP {ap:.4f}\t'
                    'HL {hl:.4f}\t'
                    'RL {rl:.4f}\t'
                    'AUC {auc:.4f}\t'.format(
            epoch, mode=mode, batch_time=batch_time,
            ap=evaluation_results[0],
            hl=evaluation_results[1],
            rl=evaluation_results[2],
            auc=evaluation_results[3]
        ))
    return evaluation_results


def main(args, file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset + '_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset + '_six_view_MaskRatios_' + str(
        args.mask_view_ratio) + '_LabelMaskRatio_' +
                              str(args.mask_label_ratio) + '_TraindataRatio_' +
                              str(args.training_sample_ratio) + '.mat')
    folds_num = args.folds_num
    folds_results = [AverageMeter() for _ in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir, args.name + "main_" + args.dataset + '_V_' + str(
            args.mask_view_ratio) + '_L_' +
                           str(args.mask_label_ratio) + '_T_' +
                           str(args.training_sample_ratio) + '_beta_' +
                           str(args.beta) + '_dropout_' +
                           str(args.dropout) + '_d_model_' +
                           str(args.d_emb) + '.txt')
    else:
        logfile = None
    logger = utils.setLogger(logfile)

    for fold_idx in range(folds_num):
        fold_idx = fold_idx
        train_dataloder, train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,
                                                                    training_ratio=args.training_sample_ratio,
                                                                    fold_idx=fold_idx,
                                                                    mode='train',
                                                                    batch_size=args.batch_size,
                                                                    shuffle=False,
                                                                    num_workers=args.workers)
        test_dataloder, test_dataset = MLdataset.getIncDataloader(data_path,
                                                                  fold_data_path,
                                                                  training_ratio=args.training_sample_ratio,
                                                                  val_ratio=0.15,
                                                                  fold_idx=fold_idx,
                                                                  mode='test',
                                                                  batch_size=args.batch_size,
                                                                  num_workers=args.workers)
        val_dataloder, val_dataset = MLdataset.getIncDataloader(data_path,
                                                                fold_data_path,
                                                                training_ratio=args.training_sample_ratio,
                                                                fold_idx=fold_idx,
                                                                mode='val',
                                                                batch_size=args.batch_size,
                                                                num_workers=args.workers)
        d_list = train_dataset.d_list  # dimension list
        n_cls = train_dataset.classes_num

        model_first = Model(d_list, args.d_emb, args.n_enc_layer, args.n_dec_layer, n_cls, args.beta, args.dropout)
        model_second = Model(d_list, args.d_emb, args.n_enc_layer, args.n_dec_layer, n_cls, args.beta, args.dropout)
        model_first = model_first.to(device)
        model_second = model_second.to(device)
        initialize(model_first)
        initialize(model_second)
        loss_first = Loss(args.alpha, args.gamma).to(device)
        loss_second = Loss(args.alpha, args.gamma).to(device)
        optimizer_first = AdamW(model_first.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_second = AdamW(model_second.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None

        best_result = 0
        best_epoch = 0
        best_first_model_dict = {"model": model_first.state_dict(), "epoch": 0}
        best_second_model_dict = {"model": model_second.state_dict(), "epoch": 0}

        for epoch in range(args.epochs):
            train_losses_first, model_first, model_second = train_first(train_dataloder, model_first, model_second,
                                                                        loss_first, loss_second
                                                                        , optimizer_first, optimizer_second, scheduler,
                                                                        epoch, logger, args.beta)
            _ = _test_first(train_dataloder, model_first, model_second, epoch, logger, "train")
            val_metric = _test_first(val_dataloder, model_first, model_second, epoch, logger, "test")

            val_metric = val_metric[0] * 0.2 + val_metric[1] * 0.2 + val_metric[2] * 0.2 + val_metric[3] * 0.4

            if val_metric > best_result:
                best_result = val_metric
                best_first_model_dict['model'] = copy.deepcopy(model_first.state_dict())
                best_first_model_dict['epoch'] = epoch
                best_second_model_dict['model'] = copy.deepcopy(model_second.state_dict())
                best_second_model_dict['epoch'] = epoch
                best_epoch = epoch

            if epoch > 50 and (epoch - best_epoch > args.patience):
                print('Training stopped: epoch=%d' % (epoch))
                break

        model_first.load_state_dict(best_first_model_dict['model'])
        model_second.load_state_dict(best_second_model_dict['model'])
        test_result_first = _test_first(test_dataloder, model_first, model_second, -1, logger, mode='test')

        logger.info(
            'final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(
                fold_idx, best_epoch, test_result_first[0], test_result_first[1], test_result_first[2],
                test_result_first[3]))

        for i in range(9):
            folds_results[i].update(test_result_first[i])
        break

    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP  HL  RL  AUCme  one_error  coverage  macAUC  macro_f1  micro_f1  alpha  beta  gamma\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg, 4)) + '+' + str(round(res.std, 4)) for res in folds_results]
    res_list.extend([str(args.alpha), str(args.beta), str(args.gamma)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write(' ' + os.path.basename(__file__))
    file_handle.write('\n')
    file_handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'records'))
    parser.add_argument('--root-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--weights_dir', type=str, metavar='PATH', default=osp.join(working_dir, 'weights'))
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--name', type=str, default='final_')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--d_emb', type=int, default=1024)
    parser.add_argument('--n_enc_layer', type=int, default=1)
    parser.add_argument('--n_dec_layer', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=5)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=30)

    args = parser.parse_args()
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    assert torch.cuda.is_available()
    device = torch.device('cuda:0')

    # hyperparams for pascal07
    lr_list = [0.0001]
    alpha_list = [500]
    beta_list = [5e-1]
    gamma_list = [50]
    d_emb_list = [768]

    args.datasets = ['pascal07']  # pascal07

    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for d_emb in d_emb_list:
                        args.d_emb = d_emb
                        for dataset in args.datasets:
                            args.dataset = dataset
                            file_path = osp.join(args.records_dir, args.name + args.dataset + '_ViewMask_' + str(
                                args.mask_view_ratio) + '_LabelMask_' + str(args.mask_label_ratio) + '_Training_' +
                                                 str(args.training_sample_ratio) + '.txt')
                            main(args, file_path)
