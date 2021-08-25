import time
# Boosting
max_depth     = 4
min_size      = 1

# import the toy model
import random
random.seed(0)
model = "exponential"
import toy_models as models
model = getattr( models, model )

# Produce training data set
nTraining     = 1000000
features, weights, diff_weights = model.get_dataset( nTraining )
import numpy as np
from Node import Node

fraction_alpha = 0.0001

if fraction_alpha>0 and fraction_alpha<1:
    bool_arr = np.zeros(len(features), dtype=bool)
    bool_arr[:int(round(fraction_alpha*len(features)))] = True
    random.shuffle(bool_arr)
    diff_weights[bool_arr]  = diff_weights[bool_arr]/fraction_alpha
    diff_weights[~bool_arr] = 0.


tic = time.time()
n1 = Node( features, max_depth, min_size, weights, diff_weights, split_method="vectorized_split_and_weight_sums")
toc = time.time()
print("vectorized split in {time:0.4f} seconds".format(time=toc-tic))

tic = time.time()
n2 = Node( features, max_depth, min_size, weights, diff_weights, split_method="iterative_split_and_weight_sums", max_relative_score_uncertainty=0.01)
toc = time.time()
print("iterative split in {time:0.4f} seconds".format(time=toc-tic))
