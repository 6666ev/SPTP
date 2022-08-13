import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics import ndcg_score


def diff2score(x):
    score = 2 * torch.exp( - x ** 2 / 500) -1
    return score

def log_square_loss(output, label):
    output = output.squeeze()
    label = label.squeeze()
    return torch.mean((torch.log(torch.clamp(output, 0, 450) + 1) - torch.log(torch.clamp(label, 0, 450) + 1)) ** 2)


def log_distance_accuracy_function(output, label):
    # 128：batch size
    # 450应该是最大刑期37年，将outputs限幅到0~37年
    output = output.squeeze()
    label = label.squeeze()
    return float(torch.mean(torch.log(torch.abs(torch.clamp(output, 0, 450) - torch.clamp(label, 0, 450)) + 1)))

def similar_score_loss(output, label):
    mc_penalty = label["mc_penalty"]
    sc_penalty = label["sc_penalty"]
    mc_penalty = torch.broadcast_to(mc_penalty.unsqueeze(1), sc_penalty.shape)

    x = mc_penalty - sc_penalty
    gt_similar_score = diff2score(x)
    return nn.L1Loss()(output, gt_similar_score)

def spearman_correlation(output, label):
    mc_penalty = label["mc_penalty"]
    sc_penalty = label["sc_penalty"]
    mc_penalty = torch.broadcast_to(mc_penalty.unsqueeze(1), sc_penalty.shape)
    gt_similar_score = diff2score(mc_penalty - sc_penalty)

    output = output.detach().cpu().numpy()
    gt_similar_score = gt_similar_score.detach().cpu().numpy()
    batch_size = output.shape[0]
    
    corr = 0
    for i in range(batch_size):
        eps = 1e-6
        gt_similar_score[i][0] -= eps
        output[i][0] -= eps
        cur_corr = stats.spearmanr(output[i] , gt_similar_score[i]).correlation
        if np.isnan(cur_corr):
            print(cur_corr)
        corr += cur_corr
    
    return corr / batch_size

def acc25(output, label):
    output = output.squeeze()
    output = output.round()
    label = label.squeeze()
    return int(torch.sum(torch.abs(output-label)/label < 0.25))/len(output)

def NDCG_at_k(output, label, K= 5):
    mc_penalty = label["mc_penalty"]
    sc_penalty = label["sc_penalty"]
    mc_penalty = torch.broadcast_to(mc_penalty.unsqueeze(1), sc_penalty.shape)
    gt_similar_score = diff2score(mc_penalty - sc_penalty)

    output = output.detach().cpu().numpy()
    gt_similar_score = gt_similar_score.detach().cpu().numpy()
    batch_size = output.shape[0]
    
    ndcg = 0
    for i in range(batch_size):
        cur_ndcg = ndcg_score([output[i]] , [gt_similar_score[i]], k=K)
        ndcg += cur_ndcg
    
    return ndcg / batch_size

def P_at_k(output, label, K= 5):
    mc_penalty = label["mc_penalty"]
    sc_penalty = label["sc_penalty"]
    mc_penalty = torch.broadcast_to(mc_penalty.unsqueeze(1), sc_penalty.shape)
    gt_similar_score = 1 / (torch.abs(mc_penalty - sc_penalty) + 1)

    output = output.detach().cpu().numpy()
    gt_similar_score = gt_similar_score.detach().cpu().numpy()
    batch_size = output.shape[0]
    
    output = np.argsort(output)[::-1]
    gt_similar_score = np.argsort(gt_similar_score)[::-1]

    p_avg = 0
    for i in range(batch_size):
        out_st = set(output[i][:K])
        gt_st = set(gt_similar_score[i][:K])
        cur_p = len(out_st & gt_st) / K
        p_avg += cur_p
    
    return p_avg / batch_size


def log_dis_np(output, label):
    return float(np.mean(np.log(np.abs(np.clip(output, 0, 450) - np.clip(label, 0, 450)) + 1)))


def acc25_np(output, label):
    return int(np.sum(np.abs(output-label)/label < 0.25))/len(output)

    