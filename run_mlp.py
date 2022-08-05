#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: run_mlp.py
@time: 2022/7/18
"""
import logging

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import BinaryDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.hidden_layer = nn.Linear(in_features=hidden_size, out_features=1)
        self.bn = nn.BatchNorm1d(num_features=1)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.input_layer.weight.data)
        nn.init.xavier_normal_(self.hidden_layer.weight.data)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.bn(x)
        output = self.sigmoid(x)

        return output


def binary_classification():
    num_epoch = 128
    input_size = 60
    hidden_size = 32
    learning_rate = 0.001
    batch_size = 32

    # data
    data_train = DataLoader(BinaryDataset(training=True), batch_size=batch_size, shuffle=True)
    data_test = DataLoader(BinaryDataset(training=False), batch_size=len(BinaryDataset(training=False)))

    # model
    model = MLP(input_size, hidden_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # start training
    for epoch in range(num_epoch):
        model.train()
        for step, (x_train, y_train) in enumerate(data_train):
            optimizer.zero_grad()
            p_train = model(x_train).view(-1)
            loss = criterion(p_train, y_train)
            loss.backward()
            optimizer.step()
            logger.info(f"step: {step} loss: {loss}")

        model.eval()
        with torch.no_grad():
            for x_test, y_test in data_test:
                p_test = model(x_test).view(-1)
                test_auc = roc_auc_score(y_test, p_test)
                logger.info(f"test: auc {test_auc}")


if __name__ == '__main__':
    binary_classification()
