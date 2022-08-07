#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: eda.py
@time: 2022/8/6
"""
import logging
from typing import List

import numpy as np
import pandas as pd
import toad
from scipy.stats import ttest_ind

from dataset import BinaryDataset

pd.set_option("display.max.columns", 10)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)
logger = logging.getLogger(__name__)


def compare_train_test(train: np.ndarray, test: np.ndarray, vocab: List[str]):
    data = np.concatenate((train, test), axis=0)

    res = pd.DataFrame(data=vocab, columns=["feature"])

    data_mean = np.mean(data, axis=0).reshape(-1, 1)
    train_mean = np.mean(train, axis=0).reshape(-1, 1)
    test_mean = np.mean(test, axis=0).reshape(-1, 1)
    data_min = np.min(data, axis=0).reshape(-1, 1)
    train_min = np.min(train, axis=0).reshape(-1, 1)
    test_min = np.min(test, axis=0).reshape(-1, 1)
    data_max = np.max(data, axis=0).reshape(-1, 1)
    train_max = np.max(train, axis=0).reshape(-1, 1)
    test_max = np.max(test, axis=0).reshape(-1, 1)
    res["All patients"] = [f"{np.float(mean_):.2f}({np.float(min_):.2f}-{np.float(max_):.2f})" for mean_, min_, max_ in
                           zip(data_mean, data_min, data_max)]
    res["Training cohort"] = [f"{np.float(mean_):.2f}({np.float(min_):.2f}-{np.float(max_):.2f})" for mean_, min_, max_
                              in zip(train_mean, train_min, train_max)]
    res["Test cohort"] = [f"{np.float(mean_):.2f}({np.float(min_):.2f}-{np.float(max_):.2f})" for mean_, min_, max_ in
                          zip(test_mean, test_min, test_max)]

    ttest_result = ttest_ind(train, test, equal_var=False, random_state=np.random.default_rng())
    res["pvalue"] = ttest_result[1]

    return res


def examine_data(data: pd.DataFrame, target: str):
    feature_names = data.drop(target, axis=1).columns
    res = pd.DataFrame(data=feature_names, columns=["feature"])

    # 计算缺失率
    res["empty_cnt"] = data.drop(target, axis=1).isnull().sum().values
    res["total_cnt"] = data.drop(target, axis=1).count(axis=0).values
    res["empty_ratio"] = res["empty_cnt"] / res["total_cnt"]

    # 计算IV
    res["iv"] = toad.quality(data, target=target, indicators=['iv']).values

    # 计算相关性
    res["corr"] = data.corr()[target].drop(target, axis=0).values

    return res[["feature", "empty_ratio", "iv", "corr"]]


def main():
    train_dataset = BinaryDataset(training=True)
    test_dataset = BinaryDataset(training=False)
    x_train, _ = train_dataset.data
    x_test, _ = test_dataset.data
    vocab = train_dataset.vocab
    r = compare_train_test(x_train, x_test, vocab)
    print(r)

    file_path = "./data/data.csv"
    data = pd.read_csv(file_path)
    to_drop = ["记帐号", "姓名", "科室", "出生日期", "手术日期", "肿块大小（彩超/CT）", "病理结果", "备注", "n麻醉费", "费用", "分级"]
    data.drop(to_drop, inplace=True, axis=1)
    examine_data(data=data.drop("分期", axis=1), target="恶性")
    examine_data(data=data.drop("恶性", axis=1), target="分期")


if __name__ == '__main__':
    main()
