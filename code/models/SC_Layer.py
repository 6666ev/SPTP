import torch
import torch.nn as nn

import torch.nn as nn
import torch
from utils.tokenizer import MyTokenizer
import numpy as np


class SC_layer(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None, mode = "base") -> None:
        super().__init__()
        self.mode = mode
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        self.hid_dim = hid_dim

        self.tokenizer = MyTokenizer(
            embedding_path="code/gensim_train/word2vec.model")
        vectors = self.tokenizer.load_embedding()
        vectors = torch.Tensor(vectors)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(vectors)

        self.lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True, dropout=0.5)

        self.hid_dim *=2 
        self.fc_penalty = nn.Linear(self.hid_dim , 1)
        self.multi_head_attn = nn.MultiheadAttention(self.hid_dim, num_heads=8, batch_first=True)

        self.inter = nn.Linear(self.hid_dim * 2, 1)
        self.dropout = nn.Dropout(0.4)


    def enc(self, case):
        text = case["fact"].cuda()
        x = self.embedding(text)
        hiddens,_ = self.lstm(x)
        out = hiddens.mean(dim=1)
        return out

    def get_similar_score_inter(self, q_emb, sc_emb):
        q_emb = q_emb.squeeze()
        sc_num = sc_emb.shape[1]

        res = []
        for i in range(sc_num):
            cur_sc_emb = sc_emb[:, i, :]
            merged_emb = torch.cat([q_emb, cur_sc_emb], dim=1)
            score = self.inter(merged_emb)
            score = nn.Sigmoid()(score)
            res.append(score)
        res = torch.cat(res, dim=1)
        return res

    def get_similar_score_cos(self, q_emb, sc_emb):
        q_emb = q_emb.squeeze()
        sc_num = sc_emb.shape[1]

        res = []
        for i in range(sc_num):
            cur_sc_emb = sc_emb[:, i, :]
            score = nn.CosineSimilarity()(q_emb, cur_sc_emb)
            res.append(score.unsqueeze(-1))
        res = torch.cat(res, dim=1)
        return res

    def get_gt_similar_score(self, mc_penalty, sc_penalty):
        mc_penalty = torch.broadcast_to(
            mc_penalty.unsqueeze(1), sc_penalty.shape)
        x = mc_penalty - sc_penalty
        # gt_similar_score = 1 / (torch.abs(x) + 1)
        gt_similar_score = 2 * torch.exp(- x ** 2 / 500) - 1
        return gt_similar_score
    
    def get_sc(self, data):
        sc_emb = torch.cat([self.enc(sc_input).unsqueeze(1)
                           for sc_input in data["sc"]], dim=1)

        # ??????similar score
        similar_score = self.get_similar_score_cos(q_emb, sc_emb)

        mc_penalty = data["query"]["penalty"].cuda()
        sc_penalty = torch.cat([sc_input["penalty"].unsqueeze(1)
                               for sc_input in data["sc"]], dim=1).cuda()
        # similar_score = self.get_gt_similar_score(mc_penalty, sc_penalty)

        # ??????top k
        _, topk_idx = torch.topk(similar_score, 5)

        to_shape = (topk_idx.shape[0], topk_idx.shape[1], sc_emb.shape[-1])
        broadcast_topk_idx = torch.broadcast_to(
            topk_idx.unsqueeze(-1), to_shape)

        sc_gathered = torch.gather(sc_emb, 1, broadcast_topk_idx)

        q_emb = self.dropout(q_emb)
        sc_gathered = self.dropout(sc_gathered)

        rerank_sc_penalty = torch.gather(sc_penalty, 1, topk_idx)

        

        # ??????attention
        _, att_mat = self.multi_head_attn(q_emb, sc_gathered, sc_gathered)
        rerank_sc_penalty = rerank_sc_penalty.unsqueeze(-1)
        rerank_sc_penalty_weighted_sum = torch.matmul(att_mat, rerank_sc_penalty) # topk weighted sum
        # rerank_sc_penalty_weighted_sum = torch.mean(rerank_sc_penalty, dim=1).unsqueeze(-1) # topk avg
        # rerank_sc_penalty_weighted_sum = sc_penalty.mean(dim=1).unsqueeze(-1).unsqueeze(-1) # candidate avg

        return similar_score, rerank_sc_penalty_weighted_sum

        
    def forward(self, data):
        q_emb = self.enc(data["query"]).unsqueeze(1)

        # vanilla ???????????? penalty
        ori_penalty = self.fc_penalty(q_emb)
        
        if self.mode == "base":
            penalty = ori_penalty 
            return {
                "penalty": penalty,
            }

        else:
            similar_score, rerank_sc_penalty_weighted_sum = self.get_sc(data)
            # ??????weighted penalty
            LAMBDA = 0.3
            penalty = (1 - LAMBDA) * ori_penalty + LAMBDA * rerank_sc_penalty_weighted_sum
            penalty = penalty.flatten(start_dim=0)

            return {
                "penalty": penalty,
                "rank": similar_score
            }
