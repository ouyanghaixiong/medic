#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: test_dataset.py
@time: 2022/7/17
"""
from unittest import TestCase

from dataset import process_raw_data, BinaryDataset, MultiDataset


class Test(TestCase):
    def test_process_raw_data(self):
        process_raw_data()


class TestBinaryDataset(TestCase):
    def test_binary_classification_data(self):
        dataset = BinaryDataset(training=True)
        x_train, y_train = dataset.data
        self.assertEqual(x_train.shape, (468, 60))
        self.assertEqual(y_train.shape, (468,))
        self.assertEqual(len(dataset), 468)

        dataset = BinaryDataset(training=False)
        x_test, y_test = dataset.data
        self.assertEqual(x_test.shape, (468, 60))
        self.assertEqual(y_test.shape, (468,))
        self.assertEqual(len(dataset), 468)

    def test_multi_classification_data(self):
        dataset = MultiDataset(training=True)
        x_train, y_train = dataset.data
        self.assertEqual(x_train.shape, (248, 60))
        self.assertEqual(y_train.shape, (248,))
        self.assertEqual(len(dataset), 248)

        dataset = MultiDataset(training=False)
        x_test, y_test = dataset.data
        self.assertEqual(x_test.shape, (249, 60))
        self.assertEqual(y_test.shape, (249,))
        self.assertEqual(len(dataset), 249)
