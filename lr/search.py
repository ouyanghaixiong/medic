#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: search.py
@time: 2022/8/6
"""
from nni.experiment import Experiment

if __name__ == '__main__':
    search_space = {
        'C': {'_type': 'choice', '_value': [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]}
    }
    experiment = Experiment('local')
    experiment.config.trial_command = 'python run.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 1000
    experiment.config.trial_concurrency = 8
    max_experiment_duration = '24h'
    experiment.run(8080)
