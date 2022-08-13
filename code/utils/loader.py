from calendar import TextCalendar, c
import jieba
import re
import os
from matplotlib.pyplot import text
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer
import pickle
from tqdm import tqdm
import random
from gensim.models import Word2Vec
from utils.tokenizer import MyTokenizer
import numpy as np
import json

RANDOM_SEED = 22
torch.manual_seed(RANDOM_SEED)


def load_embedding(embedding_path="code/gensim_train/word2vec.model"):
    model = Word2Vec.load(embedding_path)


def text_cleaner(text):
    def load_stopwords(filename):
        stopwords = []
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.replace("\n", "")
                stopwords.append(line)
        return stopwords

    stop_words = load_stopwords("code/utils/stopword.txt")

    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        # newline after </p> and </div> and <h1/>...
        {r'</(div)\s*>\s*': u'\n'},
        # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        # show links instead of texts
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]

    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace("&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    text = text.replace('+', ' ').replace(',', ' ').replace(':', ' ')
    text = re.sub("([0-9]+[年月日])+", "", text)
    text = re.sub("[a-zA-Z]+", "", text)
    text = re.sub("[0-9\.]+元", "", text)
    stop_words_user = ["年", "月", "日", "时", "分", "许", "某", "甲", "乙", "丙"]
    word_tokens = jieba.cut(text)

    def str_find_list(string, words):
        for word in words:
            if string.find(word) != -1:
                return True
        return False

    text = [w for w in word_tokens if w not in stop_words if not str_find_list(w, stop_words_user)
            if len(w) >= 1 if not w.isspace()]
    return " ".join(text)


def text_cleaner2(text):
    # 替换html特殊字符
    if type(text) is float:
        return ""
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    # 换行替换为#, 空格替换为&
    text = text.replace("#", "").replace("$", "").replace("&", "")
    text = text.replace("\n", "").replace(" ", "")

    text = jieba.lcut(text)
    # text = list(text)
    return " ".join(text)

def pad_sc_idx(sc_idx):
    res_idx=[]
    max_candidate_len = max([len(sc) for sc in sc_idx])
    for sc in sc_idx:
        case = sc[-1]
        sc += [case] * max_candidate_len
        sc = sc[:max_candidate_len]
        res_idx.append(sc)
    return res_idx

class LaicDataset(Dataset):
    def __init__(self, fact, year, province, charge, article, penalty, sc_idx, total_fact, total_charge, total_penalty, c2d):
        self.fact = fact
        self.year = torch.LongTensor(year)
        self.province = torch.LongTensor(province)
        self.charge = torch.LongTensor(charge)
        self.article = torch.LongTensor(article)
        self.penalty = torch.Tensor(penalty)

        sc_idx = pad_sc_idx(sc_idx)
        self.sc_idx = np.array(sc_idx)

        self.total_fact = total_fact
        self.total_charge = total_charge
        self.total_penalty = torch.Tensor(total_penalty)

        self.c2d = c2d

    def get_one_case(self, idx):
        return {
            "fact": self.fact["input_ids"][idx],
            "year": self.year[idx],
            "charge": self.charge[idx],
            "article": self.charge[idx],
            "province": self.province[idx],
            "penalty": self.penalty[idx],
            "cdetail": self.c2d["input_ids"][self.charge[idx]],
            "sc_idx": self.sc_idx[idx]
        }

    def get_one_similar_case(self, idx):
        return {
            "fact": self.total_fact["input_ids"][idx],
            "cdetail": self.c2d["input_ids"][self.total_charge[idx]],
            "penalty": self.total_penalty[idx],
        }

    def get_similar_case(self,qid):
        sc_idxs = self.sc_idx[qid]
        return [self.get_one_similar_case(idx) for idx in sc_idxs]

    def __getitem__(self, idx):
        return {
            "query": self.get_one_case(idx),
            "sc": self.get_similar_case(idx)
        }

    def __len__(self):
        return len(self.penalty)


def get_split_data(idx, fact, year, province, charge,article, penalty, sc_idx, c2d):
    fact_cur = {
        "input_ids": fact["input_ids"][idx],
        "token_type_ids": fact["token_type_ids"][idx],
        "attention_mask": fact["attention_mask"][idx],
    }
    year_cur = pd.Series(year)[idx].tolist()
    province_cur = pd.Series(province)[idx].tolist()
    charge_cur = pd.Series(charge)[idx].tolist()
    article_cur = pd.Series(article)[idx].tolist()
    penalty_cur = pd.Series(penalty)[idx].tolist()
    sc_idx_cur = pd.Series(sc_idx)[idx].tolist()

    return LaicDataset(fact_cur, year_cur, province_cur, charge_cur,article_cur, penalty_cur, sc_idx_cur, fact, charge, penalty, c2d)


# 标签转数字id
def label2idx(label):
    st = set(label)
    lst = sorted(list(st))  # 按照字符串顺序排列
    mp_label2idx, mp_idx2label = dict(), dict()
    for i in range(len(lst)):
        mp_label2idx[lst[i]] = i
        mp_idx2label[i] = lst[i]
    return [mp_label2idx[i] for i in label], mp_label2idx, mp_idx2label


def load_data(filename, dataset_name, tokenizer, maps=None):
    df = pd.read_csv(filename, sep=",")

    pkl_path = "code/pkl/{}/train_clean.pkl".format(dataset_name)
    if not os.path.exists(pkl_path):
        path, _ = os.path.split(pkl_path)
        if not os.path.exists(path):
            os.makedirs(path)

        fact = df["justice"].tolist()
        charge = df["charge"].tolist()
        charge = [text_cleaner2(c.replace("[","").replace("]","")) for c in charge]
        province = df["province"].tolist()
        year = df["year"].tolist()

        fact = ["{} {} {} {}".format(charge[i],province[i],year[i],fact[i]) for i in range(len(year))]

        fact = tokenizer(fact, max_length=512, return_tensors="pt", padding="max_length", truncation=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(fact, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("pkl data saved: {}".format(pkl_path))
    with open(pkl_path, "rb") as f:
        fact = pickle.load(f)

    charge = df["charge"].tolist()
    charge = [c.replace("[","").replace("]","") for c in charge]
    province = df["province"].tolist()
    year = df["year"].tolist()
    article = df["article"].tolist()
    penalty = df["judge"].tolist()
    sc_idx = [[int(x) for x in sc.split(",")] for sc in df["sc_idx"].tolist()]

    ret_maps = {}
    if maps is not None:
        charge = [maps["charge2idx"][i] for i in charge]
        year = [maps["year2idx"][i] for i in year]
        article = [maps["article2idx"][i] for i in year]
        province = [maps["province2idx"][i] for i in province]
        ret_maps = maps
    else:
        charge, mp_charge2idx, mp_idx2charge = label2idx(charge)
        article, mp_article2idx, mp_idx2article = label2idx(article)
        year, mp_year2idx, mp_idx2year = label2idx(year)
        province, mp_province2idx, mp_idx2province = label2idx(province)

        ret_maps["charge2idx"] = mp_charge2idx
        ret_maps["idx2charge"] = mp_idx2charge

        ret_maps["article2idx"] = mp_article2idx
        ret_maps["idx2article"] = mp_idx2article

        ret_maps["year2idx"] = mp_year2idx
        ret_maps["idx2year"] = mp_idx2year

        ret_maps["province2idx"] = mp_province2idx
        ret_maps["idx2province"] = mp_idx2province

    # totalset = LaicDataset(fact, year, province, charge, penalty, sc_idx)

    use_ori_split = "cail" in dataset_name
    if use_ori_split :
        train_idx = df[df["split"]==0].index
        valid_idx = df[df["split"]==1].index
        test_idx = df[df["split"]==2].index
    else:
        shuffle_idx = list(range(len(year)))
        random.seed(22)
        random.shuffle(shuffle_idx)
        train_idx = shuffle_idx[:int(len(shuffle_idx)*0.8)]
        valid_idx = shuffle_idx[int(len(shuffle_idx)*0.8): int(len(shuffle_idx)*0.9)]
        test_idx = shuffle_idx[int(len(shuffle_idx)*0.9):]

    charge2details = {}

    with open("data/meta/charge_details.json") as f:
        charge2details = json.load(f)
    
    c2d = ["#"] * len(ret_maps["charge2idx"])
    for cname, detail in charge2details.items():
        if cname not in ret_maps["charge2idx"].keys():
            continue
        cid = ret_maps["charge2idx"][cname]
        c2d[cid] = detail["定义"]

    c2d = tokenizer(c2d, max_length=128, return_tensors="pt", padding="max_length", truncation=True)

    trainset = get_split_data(train_idx, fact, year, province, charge, article, penalty, sc_idx, c2d)
    validset = get_split_data(valid_idx, fact, year, province, charge,article, penalty, sc_idx, c2d)
    testset = get_split_data(test_idx, fact, year, province, charge,article, penalty, sc_idx, c2d)

    totalset = {
        "fact": np.array(fact),
        "year": np.array(year),
        "province": np.array(province),
        "charge": np.array(charge),
        "penalty": np.array(penalty),
    }

    return totalset, trainset, validset, testset, ret_maps
