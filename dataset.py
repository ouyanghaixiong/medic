#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: dataset.py
@time: 2022/7/17
"""
import logging
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch import FloatTensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)
logger = logging.getLogger(__name__)

import pandas as pd


def process_raw_data():
    benign = pd.read_csv("./data/benign.csv")
    malignant = pd.read_csv("./data/malignant.csv")
    benign["恶性"] = 0
    malignant["恶性"] = 1
    data = pd.concat([benign, malignant], axis=0)
    data.reset_index(inplace=True, drop=True)

    to_drop = ["记帐号", "姓名", "科室", "出生日期", "手术日期", "肿块大小（彩超/CT）", "病理结果", "备注", "n麻醉费", "费用", "分级"]
    dropped = data.loc[:, to_drop]
    data.drop(to_drop, inplace=True, axis=1)

    data["年龄"] = data["年龄"].map(lambda age: age.replace("Y", "")).astype(int)

    data.replace("/", np.nan, inplace=True)
    data.replace("7.58,", "7.58", inplace=True)
    data.replace("1.13.", "1.13", inplace=True)
    data.replace("<", "", inplace=True)
    data.replace("＜6", "6", inplace=True)
    data.replace("<10", "10", inplace=True)
    data.replace("＜10", "10", inplace=True)
    data.replace("＜0.5", "0.5", inplace=True)
    data.replace("<0.5", "0.5", inplace=True)
    data.replace("＞10000", "10000", inplace=True)
    data.replace("＞1000", "1000", inplace=True)
    data.replace(">10000", "10000", inplace=True)
    data.replace("<2", "2", inplace=True)
    data.replace("＜2", "2", inplace=True)
    data.replace(">12000", "12000", inplace=True)
    data.replace("＞12000", "12000", inplace=True)
    data.replace(">1500", "1500", inplace=True)
    data.replace("1(仅剔除肿物，未行分期手术)", "1", inplace=True)
    data.replace("1（仅切除左侧附件+大网膜）", "1", inplace=True)
    data.replace("1（切除全宫+双附件+大网膜，未行分期）", "1", inplace=True)
    data.replace("？", np.nan, inplace=True)
    data.replace("1（仅切除子宫+双附件，未行分期手术）", "1", inplace=True)
    data.replace("1（仅剔除囊肿）", "1", inplace=True)
    data.replace("1（仅切除右侧附件）", "1", inplace=True)
    data.replace("1（仅剔除囊肿，未行分期手术）", "1", inplace=True)
    data.replace("1（仅剔除卵巢囊肿，未行分期手术）", "1", inplace=True)
    data.replace("1（未行分期手术）", "1", inplace=True)
    data.replace("1？", "1", inplace=True)
    data.replace("不详，术中仅切除左侧附件，肿物与周边肿物粘连致密，患者家属拒绝分期手术", np.nan, inplace=True)
    data.replace("/（卵巢恶性畸胎瘤不全术后，未行分期手术）", np.nan, inplace=True)
    data.replace("1（行卵巢癌分期手术，余均未见癌）", "1", inplace=True)
    data.replace("1/？", "1", inplace=True)
    data.replace("1（仅切除附件+大网膜）", "1", inplace=True)
    data.replace("大网膜（+）", "1", inplace=True)
    data.replace("1（仅剔除囊肿，未行分期）", "1", inplace=True)
    data.replace("4？", "4", inplace=True)
    data.replace("既往史", np.nan, inplace=True)
    data.replace("87..4", "87.4", inplace=True)
    data.replace("海泰无肿瘤标志物验单", np.nan, inplace=True)
    data.replace("0.71（入院未查，后复查，2014/04/28手术，2014/04/30出结果）", "0.71", inplace=True)
    data.replace("41（入院未查，后复查，2014/04/28手术，2014/04/30出结果）", "41", inplace=True)
    data.replace("0.6（入院未查，后复查，2014/04/28手术，2014/04/30出结果）", "0.6", inplace=True)
    data.replace("＜2.0", "2.0", inplace=True)
    data.replace("146.78（入院未查，后复查，2014/04/28手术，2014/04/30出结果）", "146.78", inplace=True)
    data.replace("62（入院未查，后复查，2014/04/28手术，2014/04/30出结果）", "62", inplace=True)
    data = data.loc[~data["白细胞"].isnull(), :]
    data = data.astype(float)
    res = dropped.merge(data, left_index=True, right_index=True)
    res.to_csv("./data/data.csv", index=False)


class BinaryDataset(Dataset):
    RANDOM_STATE = np.random.default_rng()
    FILE_PATH = "./data/data.csv"

    def __init__(self, training: bool):
        self.training = training
        data = pd.read_csv(self.FILE_PATH)
        to_drop = ["记帐号", "姓名", "科室", "出生日期", "手术日期", "肿块大小（彩超/CT）", "病理结果", "备注", "n麻醉费", "费用", "分级"]
        data.drop(to_drop, inplace=True, axis=1)
        data.fillna(0, inplace=True)
        self.vocab = data.drop(["恶性", "分期"], axis=1).columns.tolist()

        label = data["恶性"].values
        features = data.drop(["恶性", "分期"], axis=1).values
        x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.5,
                                                            random_state=self.RANDOM_STATE)

        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.training:
            return self.x_train, self.y_train
        return self.x_test, self.y_test

    def __getitem__(self, index) -> T_co:
        if self.training:
            return FloatTensor(self.x_train[index]), FloatTensor(np.asarray(self.y_train[index]))
        return FloatTensor(self.x_test[index]), FloatTensor(np.asarray(self.y_test[index]))

    def __len__(self):
        if self.training:
            return self.x_train.shape[0]
        return self.x_test.shape[0]


class MultiDataset(Dataset):
    RANDOM_STATE = np.random.default_rng()
    FILE_PATH = "./data/data.csv"

    def __init__(self, training: bool):
        self.training = training
        data = pd.read_csv(self.FILE_PATH)
        to_drop = ["记帐号", "姓名", "科室", "出生日期", "手术日期", "肿块大小（彩超/CT）", "病理结果", "备注", "n麻醉费", "费用", "分级"]
        data.drop(to_drop, inplace=True, axis=1)
        data = data.loc[data["恶性"] == 1, :]
        self.vocab = data.drop(["恶性", "分期"], axis=1).columns.tolist()
        label = data["分期"].values
        features = data.drop(["恶性", "分期"], axis=1).values
        x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.5,
                                                            random_state=self.RANDOM_STATE)

        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

    @property
    def data(self):
        if self.training:
            return self.x_train, self.y_train
        return self.x_test, self.y_test

    def __getitem__(self, index) -> T_co:
        if self.training:
            return FloatTensor(self.x_train[index]), FloatTensor(np.asarray(self.y_train[index]))
        return FloatTensor(self.x_test[index]), FloatTensor(np.asarray(self.y_test[index]))

    def __len__(self):
        if self.training:
            return self.x_train.shape[0]
        return self.x_test.shape[0]
