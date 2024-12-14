import torch
import copy
import numpy as np

def get_grad_value_theta(model, name_w, weight_decay):
    for n, p in model.named_parameters():
        # print(f'para name:{n}')
        # print(f'p dev:{p.get_device()}')
        if name_w not in n:
            cur_theta_grad = p.grad.view(-1).clone()
            if weight_decay > 0:
                cur_theta_grad = cur_theta_grad + weight_decay * p.data.view(-1).clone()

            cur_theta_value = p.data.view(-1).clone()
    return  cur_theta_grad, cur_theta_value

def get_grad_value_w(model, name_w, weight_decay):
    tmp_w_list = []
    tmp_w_value_list = []

    for n, p in model.named_parameters():
        if name_w in n:
            tmp_w_list.append(p.grad.view(-1).clone())
            tmp_w_value_list.append(p.data.view(-1).clone())
    cur_w_grad = torch.cat(tmp_w_list)
    cur_w_value = torch.cat(tmp_w_value_list)
    if weight_decay > 0:
        cur_w_grad = cur_w_grad + weight_decay * cur_w_value

    return cur_w_grad, cur_w_value


def alter_theta(model, name_w, move_theta):
    for n, p in model.named_parameters():
        if name_w not in n:
            p.data.add_(-1.0 * move_theta)


def alter_w(model, name_w, move_w):
    cur_ind = 0
    for n, p in model.named_parameters():
        if name_w in n:
            if len(p.data.size()) <= 1:
                size_para = p.data.size(0)
            else:
                size_para = p.data.size(0) * p.data.size(1)

            p.data.add_(-1.0 * torch.reshape(move_w[cur_ind:cur_ind + size_para], p.data.size() ) )
            cur_ind = cur_ind + size_para


def cal_move_para(mt_para, vt_para, grad_para, beta_1, beta_2, epoch, eps):
    # print(f'dev: -- {mt_para.get_device()} -- {grad_para.get_device()}')

    mt_para = beta_1 * mt_para + (1.0 - beta_1) * grad_para
    vt_para = beta_2 * vt_para + (1.0 - beta_2) * torch.square(grad_para)
    rec_mt_para = mt_para / (1.0 - beta_1 ** (epoch + 1))

    rec_vt_para = vt_para / (1.0 - beta_2 ** (epoch + 1))
    delta_para = rec_mt_para / (torch.sqrt(rec_vt_para) + eps)

    return delta_para, mt_para, vt_para, rec_mt_para, rec_vt_para

def init_custom_optimizer(model, name_w, device):
    tmp_w_list = []
    size_theta = 0
    size_w = 0
    for n, p in model.named_parameters():
        if name_w not in n:
            size_theta = p.data.view(-1).clone().size()
        else:
            if len(p.data.clone().size()) <= 1:
                size_w = size_w + p.data.clone().size(0)
                tmp_w_list.append(p.data.view(-1).clone())
            else:
                size_w = size_w + p.data.clone().size(0) * p.data.clone().size(1)
                tmp_w_list.append(p.data.view(-1).clone())


    mt_theta = torch.zeros(size_theta, device=device)
    mt_w = torch.zeros(size_w, device=device)
    vt_theta = torch.zeros(size_theta, device=device)
    vt_w = torch.zeros(size_w, device=device)
    mt_theta_grad_norm = torch.tensor(0.0, device=device)
    mt_w_grad_norm = torch.tensor(0.0, device=device)
    mt_theta_value_norm = torch.tensor(0.0, device=device)
    mt_w_value_norm = torch.tensor(0.0, device=device)

    return mt_theta, vt_theta, mt_w, vt_w, mt_theta_grad_norm, mt_w_grad_norm, mt_theta_value_norm, mt_w_value_norm


def cal_new_mt(mt, new_value, args, epoch, str_para):
    if str_para == 'theta':
        tmp_beta = args.beta_norm_theta
    else:
        tmp_beta = args.beta_norm_w
    mt_para = tmp_beta * mt + (1.0 - tmp_beta) * new_value
    return mt_para
    # rec_mt_para = mt_para / (1.0 - tmp_beta ** (epoch + 1))
    # return rec_mt_para


def update_model(model, name_w, mt_theta, vt_theta, mt_w, vt_w, epoch, args,
                 para_theta_his, para_w_his, mt_theta_grad_norm, mt_w_grad_norm,
                 mt_theta_value_norm, mt_w_value_norm):

    cur_theta_grad, cur_theta_value = get_grad_value_theta(model, name_w, 0.0)
    cur_w_grad, cur_w_value = get_grad_value_w(model, name_w, 0.0)
    para_theta_his.append( [torch.norm(cur_theta_grad), torch.norm(cur_theta_value)] )
    para_w_his.append( [torch.norm(cur_w_grad), torch.norm(cur_w_value)] )

    if args.net in ['ChebNetII','ChebBase']:
        cur_theta_grad = cur_theta_grad + args.prop_wd * cur_theta_value
        cur_w_grad = cur_w_grad + args.weight_decay * cur_w_value
    elif args.net in ['BernNet']:
        cur_theta_grad = cur_theta_grad + args.weight_decay * cur_theta_value
        cur_w_grad = cur_w_grad + args.weight_decay * cur_w_value
    elif args.net == 'PolyNet':# and args.base == 'jacobi':
        cur_theta_grad = cur_theta_grad + args.wd3 * cur_theta_value
        cur_w_grad = cur_w_grad + args.wd1 * cur_w_value
    else:
        cur_theta_grad = cur_theta_grad + args.weight_decay * cur_theta_value
        cur_w_grad = cur_w_grad + args.weight_decay * cur_w_value

    tmp_mt_theta = mt_theta
    tmp_vt_theta = vt_theta

    tmp_mt_w = mt_w
    tmp_vt_w = vt_w

    delta_theta, _, _, _, _ = cal_move_para(tmp_mt_theta, tmp_vt_theta, cur_theta_grad, args.beta_1_theta, args.beta_2_theta, epoch, 1e-8)
    delta_w, _, _, _, _ = cal_move_para(tmp_mt_w, tmp_vt_w, cur_w_grad, args.beta_1, args.beta_2, epoch, 1e-8 )


    # mt_theta_grad_norm = cal_new_mt(mt_theta_grad_norm, torch.norm(cur_theta_grad), args, epoch, 'theta')
    # mt_w_grad_norm = cal_new_mt(mt_w_grad_norm, torch.norm(cur_w_grad), args, epoch, 'w')
    mt_theta_value_norm = cal_new_mt(mt_theta_value_norm, torch.norm(cur_theta_value), args, epoch, 'theta')
    mt_w_value_norm = cal_new_mt(mt_w_value_norm, torch.norm(cur_w_value), args, epoch, 'w')

    if len(para_theta_his) >  args.opt_step:

        scale_theta = mt_theta_value_norm / (torch.norm(delta_theta) + 1e-8)
        scale_w = mt_w_value_norm / (torch.norm(delta_w) + 1e-8)
        cur_theta_grad = scale_theta * cur_theta_grad
        cur_w_grad = scale_w * cur_w_grad
        print(f'scale_theta:{scale_theta} -- w:{scale_w}')
    #
    #     scale_theta = mt_theta_value_norm / (mt_theta_grad_norm + 1e-8)
    #     scale_w = mt_w_value_norm / (mt_w_grad_norm + 1e-8)
    #     cur_theta_grad = scale_theta * cur_theta_grad
    #     cur_w_grad = scale_w * cur_w_grad
    #     print(f'norm theta grad:{torch.norm(cur_theta_grad) } -- w:{torch.norm(cur_w_grad)}')
    #     print(f'norm theta mt grad:{mt_theta_grad_norm } -- w:{ mt_w_grad_norm }')
    #     print(f'norm theta value:{torch.norm(cur_theta_value) } -- w:{torch.norm(cur_w_value)}')
    #     print(f'norm theta mt value:{mt_theta_value_norm} -- w:{mt_w_value_norm}')
    #



    delta_theta, mt_theta, vt_theta, rec_mt_theta, rec_vt_theta = cal_move_para(mt_theta, vt_theta, cur_theta_grad, args.beta_1_theta, args.beta_2_theta, epoch, 1e-8)
    delta_w, mt_w, vt_w, rec_mt_w, rec_vt_w = cal_move_para(mt_w, vt_w, cur_w_grad, args.beta_1, args.beta_2, epoch, 1e-8 )


    if args.net in ['ChebNetII', 'ChebBase']:
        lr_theta =  args.prop_lr
        lr_w = args.lr
    elif args.net in ['APPNP', 'GPRGNN']:
        lr_theta = args.lr
        lr_w = args.lr
    elif args.net == 'BernNet':
        lr_theta = args.Bern_lr
        lr_w = args.lr
    elif args.net == 'PolyNet':# and args.base == 'jacobi':
        lr_theta = args.lr3
        lr_w = args.lr1
    else:
        lr_theta = args.lr
        lr_w = args.lr

    # print(f'lr theta:{lr_theta} -- {lr_w}')
    alter_theta(model, name_w, lr_theta * delta_theta)
    alter_w(model, name_w, lr_w * delta_w)

    if args.ret_logit:
        return mt_theta, vt_theta, mt_w, vt_w
    else:
        # return  mt_theta, vt_theta, mt_w, vt_w, para_theta_his, para_w_his, cur_lr_theta, cur_lr_w
        return  mt_theta, vt_theta, mt_w, vt_w, para_theta_his, para_w_his, mt_theta_grad_norm, mt_w_grad_norm, mt_theta_value_norm, mt_w_value_norm
