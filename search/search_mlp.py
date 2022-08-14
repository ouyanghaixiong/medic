"""
@author: bearouyang
@contact: bearouyang@tencent.com
@file: search_dt.py
@time: 2022/8/6
"""
from nni.experiment import Experiment

if __name__ == '__main__':
    search_space = {
        'hidden_size': {'_type': 'choice', '_value': [2, 4, 6, 8, 16, 32, 64]},
        'batch_size': {'_type': 'choice', '_value': [8, 16, 32, 64, 128]},
        'weight_decay': {'_type': 'choice', '_value': [1e-3, 1e-2, 1e-1, 1.0, 10, 100]},
        'learning_rate': {'_type': 'choice', '_value': [3e-3, 3e-2, 3e-1]}
    }
    experiment = Experiment('local')
    experiment.config.trial_command = 'python run_dt.py'
    experiment.config.trial_code_directory = './trials/'
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 1000
    experiment.config.trial_concurrency = 8
    max_experiment_duration = '24h'
    experiment.run(8080)
