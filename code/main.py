from urllib.request import AbstractBasicAuthHandler
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from utils.tokenizer import MyTokenizer
import jieba

from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import numpy as np
import os
from models import TextCNN, Electra, Electra_base, LSTM, Transformer, Transformer_nj, TopJudge
from utils import loader
import argparse
import time
from torch.utils.tensorboard import SummaryWriter   
from utils.loss import log_distance_accuracy_function, log_square_loss, acc25, similar_score_loss, spearman_correlation, P_at_k, NDCG_at_k

RANDOM_SEED = 23
torch.manual_seed(RANDOM_SEED)

name2model = {
    "TextCNN": TextCNN,
    "LSTM": LSTM,
    "Transformer":Transformer,
}

name2tokenizer = {
    "TextCNN": MyTokenizer(embedding_path="code/gensim_train/word2vec.model"),
    "LSTM": MyTokenizer(embedding_path="code/gensim_train/word2vec.model"),
    "Transformer": MyTokenizer(embedding_path="code/gensim_train/word2vec.model"),
}

name2dim = {
    "TextCNN": 300,
    "LSTM": 300,
    "Transformer": 300,
}


class Trainer:
    def __init__(self, args):

        self.tokenizer = name2tokenizer[args.model_name]
        dataset_name = args.data_name

        data_path = "data/{}.csv".format(dataset_name)
        print("当前数据集路径: ", data_path)

        self.totalset, self.trainset, self.validset, self.testset, self.maps = loader.load_data(
            data_path, dataset_name, self.tokenizer)

        self.batch = int(args.batch_size)
        self.epoch = 50
        self.seq_len = 512
        self.hid_dim = 256
        self.emb_dim = name2dim[args.model_name]

        self.train_dataloader = DataLoader(dataset=self.trainset,
                                           batch_size=self.batch,
                                           shuffle=True,
                                           drop_last=False,)

        self.valid_dataloader = DataLoader(dataset=self.validset,
                                           batch_size=self.batch,
                                           shuffle=False,
                                           drop_last=False,)

        self.model = name2model[args.model_name](
            vocab_size=self.tokenizer.vocab_size, emb_dim=self.emb_dim, hid_dim=self.hid_dim, maps=self.maps, mode = args.mode)

        self.cur_time = time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        print("time: ", self.cur_time)
        self.model_name = "{}_{}".format(args.model_name, dataset_name)

        self.model_save_dir = "code/logs/{}/{}/".format(self.model_name, self.cur_time)

        if args.mode == "sc":
            self.task_name = ["penalty", "rank"]
        elif args.mode == "base":
            self.task_name = ["penalty"]

        print(self.model)
        print("train samples: ", len(self.trainset))
        print("valid samples: ", len(self.validset))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.loss_function = {
            "article": nn.CrossEntropyLoss(),
            "charge": nn.CrossEntropyLoss(),
            "penalty": log_square_loss,
            "rank": similar_score_loss,
        }

        self.score_function = {
            "article": self.f1_score_micro,
            "charge": self.f1_score_micro,
            "penalty": acc25,
            "rank": P_at_k,
        }

        if args.load_path is not None:
            print("--- stage2 ---")
            print("load model path:", args.load_path)
            checkpoint = torch.load(args.load_path)
            self.model = checkpoint['model']
            self.optimizer = checkpoint['optimizer']
            self.evaluate(args.load_path, save_result=False, evaluate_self_testset = True)
            self.set_param_trainable(trainable=True)
            # self.model = self.model.module
        print("parameter counts: ", self.count_parameters())

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            # self.model = nn.DataParallel(self.model)

    def set_param_trainable(self, trainable):
        for name, param in self.model.named_parameters():
            if "judge" not in name:
                param.requires_grad = trainable

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def f1_score_micro(self, out, label):
        return f1_score(out.cpu().argmax(1), label.cpu(), average='micro')

    def f1_score_macro(self, out, label):
        return f1_score(out.cpu().argmax(1), label.cpu(), average='macro')

    def train(self):
        best_score = -1
        
        writer = SummaryWriter(self.model_save_dir)
        global_step = 0
        for e in range(self.epoch):
            # train
            print("--- train ---")
            tq = tqdm(self.train_dataloader)
            for data in tq:
                self.optimizer.zero_grad()
                out = self.model(data)

                loss = 0
                label = {
                    "charge": data["query"]["charge"].cuda(),
                    "article": data["query"]["article"].cuda(),
                    "penalty": data["query"]["penalty"].cuda(),
                    "rank": {
                        "mc_penalty": data["query"]["penalty"].cuda(),
                        "sc_penalty": torch.cat([sc["penalty"].unsqueeze(1) for sc in data["sc"]], dim=1).cuda()
                    }
                }

                part_loss = {}
                for name in self.task_name:
                    cur_loss = self.loss_function[name](out[name], label[name])
                    part_loss[name] = np.around(
                        cur_loss.cpu().detach().numpy(), 4)
                    if name in ["rank"]:
                        cur_loss *= 10
                    loss += cur_loss

                score = {}
                for name in self.task_name:
                    score[name] = self.score_function[name](
                        out[name], label[name])

                loss.backward()
                # tq.set_postfix(epoch=e, p_loss = part_loss["penalty"], rank_loss = part_loss["rank"])
                # tq.set_postfix(epoch=e, rank_loss = part_loss["rank"])
                tq.set_postfix(epoch=e, p_loss=part_loss["penalty"])

                if "penalty" in self.task_name:
                    writer.add_scalar("penalty_loss", part_loss["penalty"], global_step)
                    writer.add_scalar("acc25", score["penalty"], global_step)

                if "rank" in self.task_name:    
                    writer.add_scalar("rank_loss", part_loss["rank"], global_step)
                    writer.add_scalar("P@5", score["rank"], global_step)

                self.optimizer.step()
                global_step += 1

            # valid
            print("--- valid ---")
            valid_out = self.infer(self.model, self.valid_dataloader)

            name = self.task_name[0]
            cur_score = self.score_function[name](valid_out[name]["pred"],
                                                  valid_out[name]["truth"])

            if cur_score > best_score:
                best_score = cur_score
                
                if not os.path.exists(self.model_save_dir):
                    os.makedirs(self.model_save_dir)
                save_path = self.model_save_dir+"best_model.pt"
                print("best model saved!")
                torch.save({"model": self.model, "optimizer": self.optimizer}, save_path)



    def infer(self, model, data_loader):
        tq = tqdm(data_loader)
        eval_out = {k: [] for k in self.task_name}
        for data in tq:
            with torch.no_grad():
                out = model(data)
                label = {
                    "charge": data["query"]["charge"].cuda(),
                    "article": data["query"]["article"].cuda(),
                    "penalty": data["query"]["penalty"].cuda(),
                    "rank": {
                        "mc_penalty": data["query"]["penalty"].cuda(),
                        "sc_penalty": torch.cat([sc["penalty"].unsqueeze(1) for sc in data["sc"]], dim=1).cuda()
                    }
                }
                for name in self.task_name:
                    eval_out[name].append((out[name], label[name]))

        for name in eval_out.keys():
            if name in ["charge","article","penalty"]:
                pred = torch.cat([i[0] for i in eval_out[name]])
                truth = torch.cat([i[1] for i in eval_out[name]])
                eval_out[name] = {"pred": pred, "truth": truth}
            elif name in ["rank"]:
                pred = torch.cat([i[0] for i in eval_out[name]])

                mc_penalty = torch.cat([i[1]["mc_penalty"]
                                       for i in eval_out[name]])
                sc_penalty = torch.cat([i[1]["sc_penalty"]
                                       for i in eval_out[name]])
                truth = {
                    "mc_penalty": mc_penalty,
                    "sc_penalty": sc_penalty
                }
                eval_out[name] = {"pred": pred, "truth": truth}
        for name in self.task_name:
            if name in ["article", "charge"]:
                print("{} micro f1:".format(name), self.f1_score_micro(
                    eval_out[name]["pred"], eval_out[name]["truth"]))
                print("{} macro f1:".format(name), self.f1_score_macro(
                    eval_out[name]["pred"], eval_out[name]["truth"]))
            elif name in ["penalty"]:
                print("penalty log distance:", log_distance_accuracy_function(
                    eval_out[name]["pred"], eval_out[name]["truth"]))
                print("penalty acc25:", acc25(
                    eval_out[name]["pred"], eval_out[name]["truth"]))
            elif name in ["rank"]:
                print("rank spearman corr:", spearman_correlation(
                    eval_out[name]["pred"], eval_out[name]["truth"]))
                print("rank P@5:",
                      P_at_k(eval_out[name]["pred"], eval_out[name]["truth"]))
                print("rank NDCG@5:",
                      NDCG_at_k(eval_out[name]["pred"], eval_out[name]["truth"]))
        return eval_out

    def evaluate(self, load_path, save_result,  evaluate_self_testset=False):
        print("--- evaluate on testset: ---")
        testset = self.testset
        print("test samples: ", len(testset))
        test_dataloader = DataLoader(dataset=testset,
                                     batch_size=self.batch,
                                     shuffle=False,
                                     drop_last=False)

        print("--- test ---")
        print("load model path: ", load_path)
        checkpoint = torch.load(load_path)
        model = checkpoint['model']
        print(model)

        test_out = self.infer(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='gpu')
    parser.add_argument('--model_name', default='Transformer', help='model_name')
    parser.add_argument('--data_name', default='laic_sc_c10', help='data_name')
    parser.add_argument('--mode', default='sc', help='sc or base')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--load_path', default=None, help='load model path')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    trainer = Trainer(args)
    trainer.train()
    print("== eval_best_model ==")
    eval_path = trainer.model_save_dir + "best_model.pt"
    trainer.evaluate(
        eval_path,
        save_result=False,
        evaluate_self_testset=True
    )
