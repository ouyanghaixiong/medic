#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: run_dt.py
@time: 2022/8/6
"""
import os
from argparse import ArgumentParser

import joblib
import nni
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import get_logger, SAVE_DIR
from dataset import BinaryDataset

logger = get_logger(__name__)


class LRPipeline:
    def __init__(self):
        self.train_dataset = BinaryDataset(training=True)
        self.test_dataset = BinaryDataset(training=False)
        self.params = {
            "C": 0.001,
            "max_iter": 1000000
        }
        self.model = None

    def _fit(self, x: np.ndarray, y: np.ndarray):
        self.model = LogisticRegression(C=self.params["C"], max_iter=self.params["max_iter"])
        self.model.fit(x, y)

    def _evaluate(self, x: np.ndarray, y: np.ndarray):
        p = self.model.predict_proba(x)[:, 1]
        roc = roc_auc_score(y, p)

        return roc

    def search(self):
        optimized_params = nni.get_next_parameter()
        self.params.update(optimized_params)
        logger.info(f"params\n{self.params}")
        x, y = self.train_dataset.data
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
        self._fit(x_train, y_train)
        roc = self._evaluate(x_valid, y_valid)
        nni.report_final_result(roc)

    def train(self):
        x_train, y_train = self.train_dataset.data
        self._fit(x_train, y_train)

    def test(self):
        x_test, y_test = self.test_dataset.data
        roc = self._evaluate(x_test, y_test)
        logger.info(f"roc on test set {roc}")

    def predict(self, x: np.ndarray):
        return self.model.predict_proba(x)[:, 1]


def main():
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    pipeline = LRPipeline()
    if not args.train:
        pipeline.search()
    else:
        pipeline.train()
        pipeline.test()
        file_path = os.path.join(SAVE_DIR, "lr.pkl")
        joblib.dump(pipeline, file_path)
        logger.info(f"流水线保存于 {file_path}")


if __name__ == '__main__':
    logger.info("begin")
    main()
    logger.info("done")
