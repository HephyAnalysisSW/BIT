import time
# Boosting
max_depth     = 10
min_size      = 30

# import the toy model
import random
random.seed(0)

import sys
sys.path.prepend('..')
import toy_models.exponential as model

# Produce training data set
nTraining     = 10000
features, training_weights, training_diff_weights = model.get_dataset( nTraining )
import numpy as np
from Node import Node

fraction_alpha = 0.1

if fraction_alpha>0 and fraction_alpha<1:
    bool_arr = np.zeros(len(features), dtype=bool)
    bool_arr[:int(round(fraction_alpha*len(features)))] = True
    random.shuffle(bool_arr)
    training_diff_weights[bool_arr]  = training_diff_weights[bool_arr]/fraction_alpha
    training_diff_weights[~bool_arr] = 0.

tic = time.time()
n0 = Node( features, training_weights, training_diff_weights, max_depth=max_depth, min_size=min_size)
toc = time.time()
print("vectorized split in {time:0.4f} seconds".format(time=toc-tic))


#tic = time.time()
#n2 = Node( features, max_depth, min_size, training_weights, training_diff_weights, split_method="iterative_split_and_weight_sums")
#toc = time.time()
#print("iterative split in {time:0.4f} seconds".format(time=toc-tic))
#

#i_feature = 0
#max_uncertainty = 0.1
#feature_values = features[:,i_feature]
#size = len(features)
#
#split_i_feature, split_value, split_gain, split_left_group = 0, -float('inf'), 0, None
#
## stable sort uses timsort like sorted in python
##feature_sorted_indices = np.argsort(feature_values, kind='stable')
#feature_sorted_indices = np.argsort(feature_values)
#sorted_weight_sums     = np.cumsum(training_weights[feature_sorted_indices])
#sorted_diff_weight_sums= np.cumsum(training_diff_weights[feature_sorted_indices])
#
## respect min size for split
#plateau_and_split_range_mask = np.ones(size-1, dtype=np.dtype('bool'))
#if min_size > 1:
#    plateau_and_split_range_mask[0:min_size-1] = False
#    plateau_and_split_range_mask[-min_size+1:] = False
#plateau_and_split_range_mask &= (np.diff(feature_values[feature_sorted_indices]) != 0)
#plateau_and_split_range_mask = plateau_and_split_range_mask.astype(int)
#
#if max_uncertainty>0:
#    n           = len(feature_values)
#    n_arr_left  = np.arange(1,n+1).astype('float')
#    n_arr_right = n-n_arr_left
#
#    sorted_weight_squared_sums      = np.cumsum( (training_weights[feature_sorted_indices]      - np.divide(sorted_weight_sums,n_arr_left,      where=n_arr_left>0,out=np.zeros_like(n_arr_left))) **2)
#    sorted_diff_weight_squared_sums = np.cumsum( (training_diff_weights[feature_sorted_indices] - np.divide(sorted_diff_weight_sums,n_arr_left, where=n_arr_left>0,out=np.zeros_like(n_arr_left))) **2)
#
#    total_weight_squared_sums  = sorted_weight_squared_sums[-1]
#    sorted_weight_squared_sums = sorted_weight_squared_sums[0:-1]
#
#    total_diff_weight_squared_sums  = sorted_diff_weight_squared_sums[-1]
#    sorted_diff_weight_squared_sums = sorted_diff_weight_squared_sums[0:-1]
#
#    n_arr_left  = n_arr_left[0:-1]
#    n_arr_right = n_arr_right[0:-1]
#    n_fac_left  = np.divide(n_arr_left,  n_arr_left-1,  where=n_arr_left>1, out=np.zeros_like(n_arr_left-1))
#    n_fac_right = np.divide(n_arr_right, n_arr_right-1, where=n_arr_right>1,out=np.zeros_like(n_arr_right-1))
#
#    variance_lambda_left  = n_fac_left*sorted_weight_squared_sums
#    variance_lambda_right = n_fac_right*(total_weight_squared_sums-sorted_weight_squared_sums)
#    variance_diff_lambda_left  = n_fac_left*sorted_diff_weight_squared_sums
#    variance_diff_lambda_right = n_fac_right*(total_diff_weight_squared_sums-sorted_diff_weight_squared_sums)
#
##tic = time.time()
#split_left_group = features[:,split_i_feature]<=split_value if not  np.isnan(split_value) else np.ones(len(features), dtype='bool')
#total_weight_sum         = sorted_weight_sums[-1]
#total_diff_weight_sum    = sorted_diff_weight_sums[-1]
#sorted_weight_sums       = sorted_weight_sums[0:-1]
#sorted_diff_weight_sums  = sorted_diff_weight_sums[0:-1]
#fisher_information_left  = sorted_diff_weight_sums**2/sorted_weight_sums
#
#sorted_diff_weight_sums_right = total_diff_weight_sum-sorted_diff_weight_sums
#sorted_weight_sums_right = total_weight_sum-sorted_weight_sums
#fisher_information_right = sorted_diff_weight_sums_right**2/sorted_weight_sums_right 
#
#fisher_gains = fisher_information_left + fisher_information_right
#
#if max_uncertainty>0:
#    #relative_information_variance_left  = 4.*variance_lambda_left/sorted_weight_sums**2 + variance_diff_lambda_left/sorted_diff_weight_sums**2
#    #relative_information_variance_right = 4.*variance_lambda_right/(total_weight_sum-sorted_weight_sums)**2 + variance_diff_lambda_right/(total_diff_weight_sum-sorted_diff_weight_sums)**2
#
#    den_left = sorted_weight_sums*sorted_diff_weight_sums_right - sorted_diff_weight_sums*sorted_weight_sums_right
#    den_right= sorted_weight_sums_right*sorted_diff_weight_sums - sorted_diff_weight_sums_right*sorted_weight_sums
#    relative_information_variance = variance_lambda_left*(sorted_weight_sums_right/sorted_weight_sums)**2* (1./total_weight_sum + 2*sorted_diff_weight_sums/den_left )**2\
#                                  + variance_lambda_right* (sorted_weight_sums/sorted_weight_sums_right)**2* (1./total_weight_sum + 2*sorted_diff_weight_sums_right/den_right )**2\
#                                  + variance_diff_lambda_left* (2.*sorted_weight_sums_right/den_left)**2\
#                                  + variance_diff_lambda_right*(2.*sorted_weight_sums/den_right)**2
#
#    #print "n_arr_left", n_arr_left
#    #print "np.divide(n_arr_left, n_arr_left-1, where=n_arr_left>1)", np.divide(n_arr_left, n_arr_left-1, where=n_arr_left>1)
#    #print "training_weights[feature_sorted_indices]", training_weights[feature_sorted_indices[:-1]][:100]
#    #print "sorted_weight_sums", sorted_weight_sums[:100] 
#    #print "np.divide(sorted_weight_sums,n_arr_left, where=n_arr_left>1)",  np.divide(sorted_weight_sums,n_arr_left, where=n_arr_left>1)[:100]
#    #print "res", (np.divide(n_arr_left, n_arr_left-1, where=n_arr_left>1) * np.cumsum( (training_weights[feature_sorted_indices[:-1]] - np.divide(sorted_weight_sums,n_arr_left, where=n_arr_left>1)) **2 ))[:100]
#    #print 
#
#argmax_fi = np.argmax(np.nan_to_num(fisher_gains)*plateau_and_split_range_mask)
#gain      =  fisher_gains[argmax_fi]
##toc = time.time()
#
##print("vectorized split in {time:0.4f} seconds".format(time=toc-tic))
#value = feature_values[feature_sorted_indices[argmax_fi]]
#
#if gain > split_gain:
#    split_i_feature = i_feature
#    split_value     = value
#    split_gain      = gain
