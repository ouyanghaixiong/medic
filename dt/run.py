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

import joblib
import nni
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from common import get_logger, SAVE_DIR
from dataset import BinaryDataset

logger = get_logger(__name__)


class DTPipeline:
    def __init__(self):
        self.train_dataset = BinaryDataset(training=True)
        self.test_dataset = BinaryDataset(training=False)
        self.params = {
            "max_depth": 8
        }
        self.model = None

    def search(self):
        optimized_params = nni.get_next_parameter()
        self.params.update(optimized_params)
        logger.info(f"params\n{self.params}")
        x, y = self.train_dataset.data
        f = StratifiedKFold()
        p = np.zeros_like(y)
        for fold, (idx_train, idx_valid) in enumerate(f.split(x, y)):
            model = DecisionTreeClassifier(max_depth=self.params["max_depth"])
            model.fit(x[idx_train], y[idx_train])
            p[idx_valid] = model.predict_proba(x[idx_valid])[:, 1]
        roc = roc_auc_score(y, p)
        logger.info(f"roc {roc}")
        nni.report_final_result(roc)

    def train(self):
        x, y = self.train_dataset.data
        model = DecisionTreeClassifier(max_depth=self.params["max_depth"])
        model.fit(x, y)
        self.model = model

    def test(self):
        x, y = self.test_dataset.data
        p = self.model.predict_proba(x)[:, 1]
        roc = roc_auc_score(y, p)
        logger.info(f"roc on test set {roc}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    pipeline = DTPipeline()
    if not args.train:
        pipeline.search()
    else:
        pipeline.train()
        pipeline.test()
        file_path = os.path.join(SAVE_DIR, "rf.pkl")
        joblib.dump(pipeline, file_path)
        logger.info(f"流水线保存于 {file_path}")


if __name__ == '__main__':
    logger.info("begin")
    main()
    logger.info("done")
