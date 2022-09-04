#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: evaluation.py
@time: 2022/8/6
"""
import os.path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, style
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve

style.use("ggplot")
sns.set_style("whitegrid")

from common import SAVE_DIR, get_logger
from dataset import BinaryDataset
from trials.run_lr import LRPipeline
from trials.run_rf import RFPipeline
from trials.run_lgbm import LGBMBinary
from trials.run_dt import DTPipeline
from trials.run_mlp import MLPBinary

logger = get_logger(__name__)

import warnings

warnings.filterwarnings("ignore")


class Graph:
    def __init__(self):
        dataset = BinaryDataset(training=False)
        self.vocab = dataset.vocab
        self.x, self.y = dataset.data
        self.lr: LRPipeline = joblib.load(os.path.join(SAVE_DIR, "lr.pkl"))
        self.dt: DTPipeline = joblib.load(os.path.join(SAVE_DIR, "dt.pkl"))
        self.rf: RFPipeline = joblib.load(os.path.join(SAVE_DIR, "rf.pkl"))
        self.lgbm: LGBMBinary = joblib.load(os.path.join(SAVE_DIR, "lgbm.pkl"))
        self.mlp: MLPBinary = joblib.load(os.path.join(SAVE_DIR, "mlp.pkl"))

        self.models = {
            "LR": self.lr,
            "DT": self.dt,
            "RF": self.rf,
            "LGBM": self.lgbm,
            "MLP": self.mlp
        }

        self.top10_features = None
        self.top10_position = None

    def analyze(self):
        fig: Figure = plt.figure(figsize=(10, 13), dpi=300)
        gs: GridSpec = fig.add_gridspec(13, 10)

        ax: Axes = fig.add_subplot(gs[:2, :])
        self.analyze_feature_importance(ax)
        ax: Axes = fig.add_subplot(gs[3:7, :4])
        self.analyze_roc_auc_curve(ax)
        ax: Axes = fig.add_subplot(gs[3:7, 6:])
        self.analyze_corr(ax)
        # 前10个变量的箱线图
        for i, feature in enumerate(self.top10_features):
            row = 8 + (i // 5) * 3
            col = (i % 5) * 2
            ax: Axes = fig.add_subplot(gs[row: row + 2, col])
            self.analyze_boxplot(i, ax, feature)

        # 保存图片
        fig.savefig("./save/figure_1.png")

    def analyze_feature_importance(self, ax):
        tmp1 = pd.DataFrame(data={"feature": self.vocab, "importance": self.rf.model.feature_importances_})
        tmp1["model"] = "RF"
        tmp2 = pd.DataFrame(data={"feature": self.vocab, "importance": self.dt.model.feature_importances_})
        tmp2["model"] = "DT"
        tmp3 = pd.DataFrame(data={"feature": self.vocab, "importance": self.lgbm.model.feature_importance()})
        tmp3["model"] = "LGBM"
        tmp3["importance"] = tmp3["importance"] / tmp3["importance"].sum()

        fi_df = pd.concat([tmp3, tmp1, tmp2], axis=0)
        self.top10_position = tmp3["importance"].argsort()[:-11:-1].tolist()
        self.top10_features = tmp3.sort_values(by="importance", ascending=False).iloc[:10, :]["feature"].tolist()
        logger.info(self.top10_features)
        fi_df = fi_df.loc[fi_df["feature"].isin(self.top10_features), :]
        sns.barplot(x="feature", y="importance", hue="model", data=fi_df, palette="pastel", ax=ax)
        ax.set_ylabel('Relative importance')
        ax.set_title("1", fontsize=25, color="moccasin", loc="left")
        ax.set_xlabel(None)

    def analyze_roc_auc_curve(self, ax):
        for name, model in self.models.items():
            p = model.predict(self.x)
            fpr, tpr, thresholds = roc_curve(self.y, p, pos_label=1)
            ax.plot(fpr, tpr, label=f"{name}")
        ax.legend(loc='lower right')
        ax.plot([0, 1], [0, 1], '--')
        ax.set_xlim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.set_title("2", fontsize=25, color="moccasin", loc="left")

    def analyze_corr(self, ax):
        # 前10个变量的相关性
        df = pd.DataFrame(self.x[:, self.top10_position], columns=self.top10_features)
        corr = df.corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.tril_indices_from(mask)] = True
        sns.heatmap(corr, ax=ax, mask=mask, cmap=sns.diverging_palette(220, 20, n=200), center=0, square=True,
                    fmt=".2f")
        ax.set_title("3", fontsize=25, color="moccasin", loc="left")


    def analyze_boxplot(self, i, ax, feature):
        features = pd.DataFrame(data=self.x, columns=self.vocab)
        features["label"] = pd.Series(self.y).map(lambda v: "Benign" if v == 0 else "Malignant")
        sns.boxplot(x="label", y=feature, data=features, orient='v',
                    flierprops={'marker': '.', 'markerfacecolor': 'black', 'color': 'black'},
                    palette="pastel", ax=ax)
        ax.set_title(f"4-{i+1}", fontsize=25, color="moccasin", loc="left")
        ax.set_xlabel(None)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

if __name__ == '__main__':
    g = Graph()
    g.analyze()
