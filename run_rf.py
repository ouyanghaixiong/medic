#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: run_rf.py
@time: 2022/7/18
"""

import logging

from sklearn.ensemble import RandomForestClassifier
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
    x_train, y_train = BinaryDataset(training=True).data
    x_test, y_test = BinaryDataset(training=False).data
    f = StratifiedKFold()
    for no, (idx_train, idx_valid) in enumerate(f.split(x_train, y_train)):
        _model = RandomForestClassifier()
        _model.fit(x_train[idx_train], y_train[idx_train])
        p = _model.predict_proba(x_train[idx_valid])[:, 1]
        roc = roc_auc_score(y_train[idx_valid], p)
        logging.info(f"fold {no} roc {roc}")

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    p_test = model.predict_proba(x_test)[:, 1]
    roc_test = roc_auc_score(y_test, p_test)
    logging.info(f"roc_test {roc_test}")


if __name__ == '__main__':
    binary_classification()
