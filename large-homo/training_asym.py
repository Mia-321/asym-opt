import argparse
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
import math
from dataset_loader import DataLoader
from utils import random_splits_eachclass,random_splits,random_splits_citation,fixed_splits, set_seed

from models import *
from asym_opt import *

from gpr_model import GPRGNN
from bern_model import BernNet
from poly_model import PolyNet


def train(model, optimizer, data, args, name_w, epoch, mt_theta, vt_theta, mt_w, vt_w,
          para_theta_his, para_w_his, mt_theta_grad_norm, mt_w_grad_norm, mt_theta_value_norm, mt_w_value_norm):
    dprate = args.dprate
    model.train()
    # for n,p in model.named_parameters():
    #     print(f'n:{ n}--p:{p.data[0:5]}')

    optimizer.zero_grad()
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    if args.asym:
        mt_theta, vt_theta, mt_w, vt_w, para_theta_his, para_w_his, mt_theta_grad_norm, \
            mt_w_grad_norm, mt_theta_value_norm, mt_w_value_norm \
            = update_model(model, name_w, mt_theta, vt_theta,
                           mt_w, vt_w, epoch, args, para_theta_his, para_w_his,
                           mt_theta_grad_norm, mt_w_grad_norm, mt_theta_value_norm, mt_w_value_norm)
        # optimizer_grad_module.step()
    else:
        optimizer.step()

    del out

    return mt_theta, vt_theta, mt_w, vt_w, para_theta_his, para_w_his, mt_theta_grad_norm, mt_w_grad_norm, mt_theta_value_norm, mt_w_value_norm


def test(model, data):
    model.eval()
    logits, accs, losses, preds = model(data), [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.nll_loss(model(data)[mask], data.y[mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses



def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    # device = torch.cuda.set_device(args.device) if torch.cuda.is_available() else 'cpu'
    if not args.full and args.net == 'ChebNetII' and args.dataset in ['Chameleon','Squirrel']:
        Net = ChebNetII_V
    tmp_net = Net(dataset, args)
    #Using the dataset splits described in the paper.
    if args.full:
        data = random_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed)
    elif args.semi_rnd:
        if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            if args.train_rate == 0.025:
                data = random_splits_citation(data, dataset.num_classes)
            else:
                data = random_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed)
        else:
            data = random_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed)
    elif args.semi_fix and args.dataset in ["Chameleon", "Squirrel", "Actor", "Texas", "Cornell", "Wisconsin"]:
        data = fixed_splits(data, dataset.num_classes, percls_trn, val_lb, args.dataset)
    
    model, data = tmp_net.to(device), data.to(device)
    if args.net in ['ChebNetII','ChebBase']:
        optimizer = torch.optim.Adam([{ 'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': args.prop_wd, 'lr': args.prop_lr}])
    elif args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    elif args.net == 'BernNet':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
    elif args.net == 'PolyNet': # and args.base == 'jacobi':
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                      {'params': model.lin2.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                      {'params': model.comb_weight, 'weight_decay': args.wd3, 'lr': args.lr3}])
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    best_train_acc = 0
    best_test_acc = 0
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    best_step = -1

    val_loss_history = []
    val_acc_history = []

    para_theta_his = []
    para_w_his = []

    name_w = 'lin'
    mt_theta, vt_theta, mt_w, vt_w, \
        mt_theta_grad_norm, mt_w_grad_norm, mt_theta_value_norm, mt_w_value_norm \
        = init_custom_optimizer(model, name_w, device)


    time_run=[]
    for epoch in range(args.epochs):
        t_st = time.time()
        mt_theta, vt_theta, mt_w, vt_w, para_theta_his, para_w_his,\
            mt_theta_grad_norm, mt_w_grad_norm, mt_theta_value_norm, mt_w_value_norm \
            = train(model, optimizer, data, args, name_w, epoch,mt_theta, vt_theta,
                    mt_w, vt_w, para_theta_his, para_w_his,
                    mt_theta_grad_norm, mt_w_grad_norm, mt_theta_value_norm, mt_w_value_norm)

        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)
        print(f'epoch:{epoch} train_loss:{train_loss} val_loss:{val_loss} test_loss:{tmp_test_loss} '
              f'train_acc:{train_acc} val_acc:{val_acc} test_acc:{test_acc}')

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net in ['ChebNetII']:
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            elif args.net in ['ChebBase']:
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu().numpy()
            else:
                theta = args.alpha

            # add for debug
            best_train_acc = train_acc
            best_test_acc = tmp_test_acc
            best_train_loss = train_loss
            best_test_loss = tmp_test_loss
            best_step = epoch

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break


    print(f'best_step:{best_step} train_acc:{best_train_acc} test_acc:{best_test_acc} val_acc:{best_val_acc} '
          f'train_loss:{best_train_loss} test_loss:{best_test_loss} val_loss:{best_val_loss}')
    return best_step, best_train_acc, best_train_loss, best_test_acc, best_test_loss, best_val_acc, best_val_loss, theta, time_run
    # return test_acc, best_val_acc, theta, time_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--train_rate', type=float, default=0.0, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.0, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')

    parser.add_argument('--dataset', type=str, choices=['Cora','flickr', 'computers', 'photo','coauthor-cs', 'coauthor-physics'],default='Cora')
    parser.add_argument('--device', type=int, default=1, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GCN', 'APPNP', 'ChebNet','MLP','ChebNetII','ChebBase', 'GPRGNN', 'BernNet', 'PolyNet'], default='ChebNetII')
    parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    parser.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')

    parser.add_argument('--q', type=int, default=0, help='The constant for ChebBase.')
    parser.add_argument('--full', type=bool, default=False, help='full-supervise with random splits')
    parser.add_argument('--semi_rnd', type=bool, default=False, help='semi-supervised with random splits')
    parser.add_argument('--semi_fix', type=bool, default=False, help='semi-supervised with fixed splits')


    #gprgnn
    parser.add_argument('--ppnp', default='GPR_prop', choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)

    #bern
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')

    #poly
    parser.add_argument('--base', type=str, choices=['cheb', 'jacobi'], default='cheb')
    parser.add_argument('--lr1', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--lr2', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--lr3', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--wd1', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--wd2', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--wd3', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--a', type=float, default=0.0, help='a')
    parser.add_argument('--b', type=float, default=0.0, help='b')
    parser.add_argument('--c_f', type=float, default=1.0, help='cf')
    parser.add_argument('--c_y', type=float, default=1.0, help='cy')

    parser.add_argument('--asym', default=False, action='store_true')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1.')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2.')
    parser.add_argument('--beta_1_theta', type=float, default=0.9, help='beta_1_theta.')
    parser.add_argument('--beta_2_theta', type=float, default=0.999, help='beta_2_theta.')

    parser.add_argument('--inter_alpha', type=float, default=1.0, help='inter_alpha.')
    parser.add_argument('--ret_logit', default=False, action='store_true')
    parser.add_argument('--with_bias', default=True, action='store_false')
    parser.add_argument('--noise_ratio', type=float, default=0.01, help='noise ratio.')
    parser.add_argument('--keep_prob', type=float, default=0.9, help='noise ratio.')

    parser.add_argument('--para_tau', type=float, default=1.0, help='para_tau.')
    parser.add_argument('--opt_step', type=int, default=10, help='optimizer step.')
    parser.add_argument('--clip_lower', type=float, default=0.00001, help='clip_lower.')
    parser.add_argument('--clip_upper', type=float, default=1.0, help='clip_upper.')

    parser.add_argument('--beta_norm_theta', type=float, default=0.9, help='beta norm theta.')
    parser.add_argument('--beta_norm_w', type=float, default=0.9, help='beta norm w.')

    args = parser.parse_args()
    set_seed(args.seed)
    #10 fixed seeds for random splits from BernNet
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name =='MLP':
        Net = MLP
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name =='ChebBase':
        Net = ChebBase
    elif gnn_name == "ChebNetII":
        Net = ChebNetII
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name == 'PolyNet':
        Net = PolyNet
    
    dataset = DataLoader(args.dataset)
    data = dataset[0]

    # print stats of dataset
    print(f'nodes:{data.x.size(0)} edges:{data.edge_index.size(1)} num_class:{dataset.num_classes} feat dim:{dataset.num_features}')
    homo_edge_count = torch.sum(data.y[data.edge_index[0, :]] == data.y[data.edge_index[1, :]])
    print(f'homo edge ratio: {homo_edge_count * 1.0 / (data.edge_index.size(1) * 1.0)}')

    if args.full:
        args.train_rate = 0.6
        args.val_rate = 0.2
    else:
        if args.train_rate < 0.001:
            args.train_rate = 0.025
            args.val_rate = 0.025
    percls_trn = int(round(args.train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(args.val_rate*len(data.y)))
    
    # results = []
    time_results=[]
    best_step_list = []
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    for RP in tqdm(range(args.runs)):
        print(f'run:{RP}')
        args.seed=SEEDS[RP]
        # test_acc, best_val_acc, theta_0,time_run = RunExp(args, dataset, data, Net, percls_trn, val_lb)
        best_step, best_train_acc, best_train_loss, best_test_acc, best_test_loss, best_val_acc, best_val_loss, theta, time_run = RunExp(args, dataset, data, Net, percls_trn, val_lb)

        time_results.append(time_run)
        best_step_list.append(best_step)
        train_acc_list.append(best_train_acc)
        train_loss_list.append(best_train_loss)
        test_acc_list.append(best_test_acc)
        test_loss_list.append(best_test_loss)

        # # results.append([test_acc, best_val_acc, theta_0])
        # results.append([best_test_acc, best_val_acc])
        # print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')
        # if args.net in ["ChebBase","ChebNetII"]:
        #     print('Weights:', [float('{:.4f}'.format(i)) for i in theta_0])

    print(f'adNet name: {gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    for i in range(args.runs):
        print('{} : {} ---- {}'.format(i, test_acc_list[i], best_step_list[i]))
    print('{} on {}'.format(args.net, args.dataset))

    test_acc_mean = np.mean(np.array(test_acc_list))
    test_acc_std = np.max(np.abs(
        sns.utils.ci(sns.algorithms.bootstrap(np.array(test_acc_list), func=np.mean, n_boot=1000), 95) - np.array(
            test_acc_list).mean()))
    print(f'test acc mean = {test_acc_mean * 100:.4f} ± {test_acc_std * 100:.4f}')

    train_acc_mean = np.mean(np.array(train_acc_list))
    # train_acc_std = torch.sqrt(torch.var(torch.Tensor(train_acc_list)))
    train_acc_std = uncertainty = np.max(np.abs(
        sns.utils.ci(sns.algorithms.bootstrap(np.array(train_acc_list), func=np.mean, n_boot=1000), 95) - np.array(
            train_acc_list).mean()))
    print(f'train acc mean = {train_acc_mean * 100:.4f} ± {train_acc_std * 100:.4f}')

    print('train acc list:{}\ntest acc list:{}'.format(train_acc_list, test_acc_list))

    gap_acc_tensor = (np.array(train_acc_list) - np.array(test_acc_list)) * 100.0
    gap_loss_tensor = np.array(test_loss_list) - np.array(train_loss_list)
    print('gap_acc:{}'.format(gap_acc_tensor.tolist()))
    print('gap loss:{}'.format(gap_loss_tensor.tolist()))
    mean_gap_acc, std_gap_acc = np.mean(gap_acc_tensor), np.max(np.abs(
        sns.utils.ci(sns.algorithms.bootstrap(gap_acc_tensor, func=np.mean, n_boot=1000), 95) - gap_acc_tensor.mean()))
    mean_gap_loss, std_gap_loss = np.mean(gap_loss_tensor), np.max(np.abs(
        sns.utils.ci(sns.algorithms.bootstrap(gap_loss_tensor, func=np.mean, n_boot=1000),
                     95) - gap_loss_tensor.mean()))

    print(f'gap acc mean = {mean_gap_acc:.4f} ± {std_gap_acc:.4f}')
    print(f'gap loss mean = {mean_gap_loss:.4f} ± {std_gap_loss:.4f}')

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)
    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    print(f'finish')
