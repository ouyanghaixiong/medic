#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: run.py
@time: 2022/8/6
"""
import os
from argparse import ArgumentParser
from typing import List

import joblib
import lightgbm as lgb
import nni
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import get_logger, SAVE_DIR
from dataset import BinaryDataset

logger = get_logger(__name__)


class LGBMBinary:
    def __init__(self):
        self.train_dataset = BinaryDataset(training=True)
        self.test_dataset = BinaryDataset(training=False)
        self.params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'num_class': 1,
            'objective': "binary",
            'metric': "auc",
            'num_leaves': 8,
            'min_data': 10,
            'boost_from_average': True,
            'num_threads': -1,
            'feature_fraction': 0.8,
            'learning_rate': 0.001,
        }
        self.num_boost_round = 1000
        self.model = None

    def search(self):
        optimized_params = nni.get_next_parameter()
        self.params.update(optimized_params)
        logger.info(f"params\n{self.params}")
        x, y = self.train_dataset.data
        f = StratifiedKFold()
        p = np.zeros_like(y)
        for no, (idx_train, idx_valid) in enumerate(f.split(x, y)):
            train_data = lgb.Dataset(x[idx_train], y[idx_train])
            valid_data = lgb.Dataset(x[idx_valid], y[idx_valid])
            model = lgb.train(self.params,
                              train_data,
                              num_boost_round=1000,
                              early_stopping_rounds=100,
                              valid_sets=[train_data, valid_data])
            p[idx_valid] = model.predict(x[idx_valid])
        auc = roc_auc_score(y, p)
        nni.report_final_result(auc)

    def train(self):
        x, y = self.train_dataset.data
        train_data = lgb.Dataset(x, y)
        self.model = lgb.train(self.params, train_data, num_boost_round=self.num_boost_round)

    def test(self):
        x, y = self.test_dataset.data
        p = self.model.predict(x)
        auc = roc_auc_score(y, p)
        logger.info(f"roc on test set {auc}")

        vocab: List[str] = BinaryDataset(training=False).vocab
        importance = pd.Series(self.model.feature_importance(), index=vocab).sort_values(key=lambda v: -v)
        logger.info(f"feature importance\n{importance}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    pipeline = LGBMBinary()
    if not args.train:
        pipeline.search()
    else:
        pipeline.train()
        pipeline.test()
        file_path = os.path.join(SAVE_DIR, "lgbm.pkl")
        joblib.dump(pipeline, file_path)
        logger.info(f"流水线保存于 {file_path}")


if __name__ == '__main__':
    logger.info("begin")
    main()
    logger.info("done")
