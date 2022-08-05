#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: run_lightgbm.py
@time: 2022/7/18
"""
import logging
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)
logger = logging.getLogger(__name__)

from sklearn.metrics import roc_auc_score

from dataset import BinaryDataset


def binary_classification():
    hparams = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'num_class': 1,
        'objective': "binary",
        'metric': "auc",
        'num_leaves': 8,
        'min_data': 10,
        'boost_from_average': True,
        # set it according to your cpu cores.
        'num_threads': -1,
        'feature_fraction': 0.8,
        'learning_rate': 0.001,
    }
    x_train, y_train = BinaryDataset(training=True).data
    x_test, y_test = BinaryDataset(training=False).data
    f = StratifiedKFold()
    p_train = np.zeros_like(y_train)
    for no, (idx_train, idx_valid) in enumerate(f.split(x_train, y_train)):
        train_data = lgb.Dataset(x_train[idx_train], y_train[idx_train])
        valid_data = lgb.Dataset(x_train[idx_valid], y_train[idx_valid])
        model = lgb.train(hparams,
                          train_data,
                          num_boost_round=1000,
                          early_stopping_rounds=100,
                          valid_sets=[train_data, valid_data])
        p_train[idx_valid] = model.predict(x_train[idx_valid])
    auc_train = roc_auc_score(y_train, p_train)
    logger.info(f"auc_train {auc_train}")

    model = lgb.train(hparams, train_set=lgb.Dataset(x_train, y_train), num_boost_round=100)
    p_test = model.predict(x_test)
    auc_test = roc_auc_score(y_test, p_test)
    logger.info(f"auc_test {auc_test}")

    cols: List[str] = BinaryDataset(training=False).vocab
    importance = pd.Series(model.feature_importance(), index=cols).sort_values(key=lambda x: -x)
    logger.info(f"feature importance\n{importance}")


if __name__ == '__main__':
    binary_classification()
