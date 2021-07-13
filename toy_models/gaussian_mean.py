#!/usr/bin/env python
# Standard imports

import numpy as np
import random
import ROOT

# Definition of the model
## exponential
mu    = 0
sigma = 1.
xmin  = -4
xmax  = 4 
model = ROOT.TF1("model", "1./({sigma}*sqrt(2*pi))*exp( -0.5*((x-{mu})/{sigma})**2)".format(mu=mu, sigma=sigma), xmin, xmax)
score_theory     = ROOT.TF1("score_theory", "(x-{mu})/{sigma}**2".format(mu=mu, sigma=sigma), xmin, xmax)
min_score_theory = min( score_theory.Eval(xmin), score_theory.Eval(xmax) )
max_score_theory = max( score_theory.Eval(xmin), score_theory.Eval(xmax) ) 

make_log  = True
n_events      = 100000
n_trees       = 100
learning_rate = 0.2 
max_depth     = 2
min_size      = 50
n_plot = 10 # Plot every tenth
texX   = "x"
weighted      = False
id_string     = "nTrees%i-gaussian_mean"%(n_trees)
if weighted:id_+='-weighted'

def get_sampled_dataset( n_events ):
    features = np.array( [ [model.GetRandom(xmin, xmax)] for i in range(n_events)] )
    weights       = np.array( [1 for i in range(n_events)] ) 
    diff_weights  = np.array( [ (features[i][0]-mu)/sigma**2 for i in range(n_events)] )
    return features, weights, diff_weights
#def get_weighted_dataset( n_events ):
#    features = np.array( [ [xmin+random.random()*(xmax-xmin)] for i in range(n_events)] )
#    weights       = np.array( [ model.Eval(features[i][0]) for i in range(n_events)] ) 
#    diff_weights  = np.array( [ weights[i]*(1./alpha - (features[i][0]-pT0)) for i in range(n_events)] )
#    return features, weights, diff_weights

#get_dataset = get_weighted_dataset 
get_dataset = get_sampled_dataset 
