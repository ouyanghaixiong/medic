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
from sklearn.metrics import roc_curve

style.use("ggplot")
sns.set_style("whitegrid")

from common import SAVE_DIR, get_logger
from dataset import BinaryDataset
from lr.run_lr import LRPipeline
from rf.run_rf import RFPipeline
from lgbm.run_lgbm import LGBMBinary
from dt.run_dt import DTPipeline
from mlp.run_mlp import MLPBinary

logger = get_logger(__name__)


def analyze_roc_auc():
    dataset = BinaryDataset(training=False)
    x, y = dataset.data

    lr: LRPipeline = joblib.load(os.path.join(SAVE_DIR, "lr.pkl"))
    dt: DTPipeline = joblib.load(os.path.join(SAVE_DIR, "dt.pkl"))
    rf: RFPipeline = joblib.load(os.path.join(SAVE_DIR, "rf.pkl"))
    lgbm: LGBMBinary = joblib.load(os.path.join(SAVE_DIR, "lgbm.pkl"))
    mlp: MLPBinary = joblib.load(os.path.join(SAVE_DIR, "mlp.pkl"))
    models = {
        "LR": lr,
        "DT": dt,
        "RF": rf,
        "LGBM": lgbm,
        "MLP": mlp
    }

    # 画图
    fig: Figure = plt.figure(figsize=(10, 13), dpi=300)
    gs = fig.add_gridspec(13, 10)

    # 特征重要性
    vocab = dataset.vocab
    tmp1 = pd.DataFrame(data={"feature": vocab, "importance": rf.model.feature_importances_})
    tmp1["model"] = "RF"
    tmp2 = pd.DataFrame(data={"feature": vocab, "importance": dt.model.feature_importances_})
    tmp2["model"] = "DT"
    tmp3 = pd.DataFrame(data={"feature": vocab, "importance": lgbm.model.feature_importance()})
    tmp3["model"] = "LGBM"
    tmp3["importance"] = tmp3["importance"] / tmp3["importance"].sum()
    fi_df = pd.concat([tmp1, tmp2, tmp3], axis=0)
    top10_position = tmp3["importance"].argsort()[:-11:-1].tolist()
    top10_features = tmp3.sort_values(by="importance", ascending=False).iloc[:10, :]["feature"].tolist()
    fi_df = fi_df.loc[fi_df["feature"].isin(top10_features), :]
    ax: Axes = fig.add_subplot(gs[:2, :])
    sns.barplot(x="feature", y="importance", hue="model", data=fi_df, palette="pastel", ax=ax)
    ax.set_ylabel('relative importance')

    # roc曲线
    ax: Axes = fig.add_subplot(gs[3:7, :4])
    for name, model in models.items():
        p = model.predict(x)
        fpr, tpr, thresholds = roc_curve(y, p, pos_label=1)
        ax.plot(fpr, tpr, label=f"{name}")
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], '--')
    ax.set_xlim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

    # 前10个变量的相关性
    ax: Axes = fig.add_subplot(gs[3:7, 6:])
    df = pd.DataFrame(x[:, top10_position], columns=top10_features)
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.tril_indices_from(mask)] = True
    sns.heatmap(corr, ax=ax, mask=mask, cmap=sns.diverging_palette(220, 20, n=200), center=0, square=True, fmt=".2f")

    # 前10个变量的箱线图
    features = pd.DataFrame(data=x, columns=vocab)
    features["label"] = pd.Series(y).map(lambda v: "Benign" if v == 0 else "Malignant")
    for i, feature in enumerate(top10_features):
        row = 8 + (i // 5) * 3
        col = (i % 5) * 2
        ax: Axes = fig.add_subplot(gs[row: row + 2, col])
        sns.boxplot(x="label", y=feature, data=features, orient='v',
                    flierprops={'marker': '.', 'markerfacecolor': 'black', 'color': 'black'},
                    palette="pastel", ax=ax)
        ax.set_xlabel("")

    # 保存图片
    fig.savefig("./save/figure_1.png")


if __name__ == '__main__':
    analyze_roc_auc()
