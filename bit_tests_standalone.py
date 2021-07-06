#!/usr/bin/env python
# Standard imports

import numpy as np
import random
import cProfile
import time
from BoostedInformationTree import BoostedInformationTree


training_features = np.loadtxt('training_features_power_law_model.txt.gz')
training_features = training_features.reshape(training_features.shape[0], 1)
print(training_features.shape)
training_weights = np.loadtxt('training_weights_power_law_model.txt.gz')
training_diff_weights = np.loadtxt('training_diff_weights_power_law_model.txt.gz')

learning_rate = 0.02
n_trees       = 100
learning_rate = 0.2 
max_depth     = 2
min_size      = 50

bit = BoostedInformationTree(
        training_features = training_features,
        training_weights      = training_weights, 
        training_diff_weights = training_diff_weights, 
        learning_rate = learning_rate, 
        n_trees = n_trees,
        max_depth=max_depth,
        min_size=min_size,
        split_method='vectorized_split_and_weight_sums',
        weights_update_method='vectorized')

bit.boost()
