"""
@author: ouyang
@contact: ouyhaix@icoud.com
@file: run_dt.py
@time: 2022/8/7
"""

import os
from argparse import ArgumentParser

import joblib
import nni
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from common import get_logger, SAVE_DIR
from dataset import BinaryDataset

logger = get_logger(__name__)


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


class MLPBinary:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.params = {
            "hidden_size": 32,
            "batch_size": 32,
            "weight_decay": 0.01,
            "learning_rate": 0.001
        }
        self.num_epoch = 64
        self.model = None
        self.train_dataset = BinaryDataset(training=True)
        self.test_dataset = BinaryDataset(training=False)

    def search(self):
        optimized_params = nni.get_next_parameter()
        self.params.update(optimized_params)
        logger.info(f"params\n{self.params}")

        f = KFold(n_splits=5, shuffle=True, random_state=42)
        y = torch.zeros(self.train_dataset.__len__(), 1).view(-1).to(self.DEVICE)
        p = torch.zeros((self.train_dataset.__len__(), 1)).view(-1).to(self.DEVICE)
        for fold, (idx_train, idx_valid) in enumerate(f.split(self.train_dataset)):
            model = MLP(input_size=45, hidden_size=self.params["hidden_size"])
            model.to(self.DEVICE)
            criterion = nn.BCELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.params["learning_rate"],
                                          weight_decay=self.params["weight_decay"])
            train_sampler = torch.utils.data.SubsetRandomSampler(idx_train)
            valid_sampler = torch.utils.data.SubsetRandomSampler(idx_valid)
            train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params["batch_size"],
                                                       sampler=train_sampler)
            valid_loader = torch.utils.data.DataLoader(self.train_dataset, len(idx_valid), sampler=valid_sampler)
            for epoch in range(self.num_epoch):
                for step, (x_train, y_train) in enumerate(train_loader):
                    x_train = x_train.to(self.DEVICE)
                    y_train = y_train.to(self.DEVICE)
                    model.train()
                    optimizer.zero_grad()
                    p_train = model(x_train).view(-1)
                    loss = criterion(p_train, y_train)
                    loss.backward()
                    optimizer.step()
                    logger.info(f"epoch {epoch} step {step} loss {loss}")
            model.eval()
            with torch.no_grad():
                for x_valid, y_valid in valid_loader:
                    x_valid = x_valid.to(self.DEVICE)
                    y_valid = y_valid.to(self.DEVICE)
                    y[idx_valid] = y_valid
                    p[idx_valid] = model(x_valid).view(-1)
        auc = roc_auc_score(y.cpu(), p.cpu())
        logger.info(f"auc {auc}")
        nni.report_final_result(auc)

    def train(self):
        model = MLP(input_size=45, hidden_size=self.params["hidden_size"])
        model = model.to(self.DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.params["learning_rate"],
                                      weight_decay=self.params["weight_decay"])
        train_loader = DataLoader(self.train_dataset, batch_size=self.params["batch_size"], shuffle=True)
        for epoch in range(self.num_epoch):
            for step, (x, y) in enumerate(train_loader):
                x = x.to(self.DEVICE)
                y = y.to(self.DEVICE)
                model.train()
                optimizer.zero_grad()
                p = model(x).view(-1)
                loss = criterion(p, y)
                loss.backward()
                optimizer.step()
                logger.info(f"epoch {epoch} step {step} loss {loss}")
        self.model = model

    def test(self):
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=self.test_dataset.__len__())
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.DEVICE)
                y = y.to(self.DEVICE)
                p = self.model(x).view(-1)
                auc = roc_auc_score(y.cpu(), p.cpu())
                logger.info(f"auc on test dataset {auc}")

    def predict(self, x: np.ndarray):
        with torch.no_grad():
            x = torch.Tensor(x).to(self.DEVICE)
            return self.model(x).view(-1).cpu().numpy()


def main():
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    pipeline = MLPBinary()
    if not args.train:
        pipeline.search()
    else:
        pipeline.train()
        pipeline.test()
        file_path = os.path.join(SAVE_DIR, "mlp.pkl")
        joblib.dump(pipeline, file_path)
        logger.info(f"流水线保存于 {file_path}")


if __name__ == '__main__':
    logger.info("begin")
    main()
    logger.info("done")
