#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: common.py
@time: 2022/7/17
"""
import logging
import os

import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)

# the absolute root dir of the project
ROOT_DIR: str = os.path.dirname(__file__)

# the absolute data dir
DATA_DIR = os.path.join(ROOT_DIR, "data")

# the absolute save dir
SAVE_DIR = os.path.join(ROOT_DIR, "save")

# the device to use
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str):
    return logging.getLogger(name)
