#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: test_eda.py
@time: 2022/8/6
"""
from unittest import TestCase

import pandas as pd

from eda import examine_data

pd.set_option("display.max.rows", 100)


class Test(TestCase):
    def test_examine_empty(self):
        file_path = "./data/data.csv"
        data = pd.read_csv(file_path)
        to_drop = ["记帐号", "姓名", "科室", "出生日期", "手术日期", "肿块大小（彩超/CT）", "病理结果", "备注", "n麻醉费", "费用", "分级"]
        data.drop(to_drop, inplace=True, axis=1)

        res = examine_data(data.drop("分期", axis=1), target="恶性")
        print(res)
