#!/usr/bin/env python
# Standard imports

import numpy as np
import random
import ROOT
from math import sin, cos, pi

# Definition of the model
## exponential
alpha = 0
theta0 = 0 
xmin  = 0
xmax  = 2*pi 
texX  = "#phi"
model = ROOT.TF1("model", "1./(2*pi)+{theta}*sin(x)".format( theta=theta0), xmin, xmax)
score_theory     = ROOT.TF1("score_theory", "sin(x)/(1./(2*pi)+{theta}*sin(x))".format(theta=theta0), xmin, xmax)
min_score_theory = -2*pi 
max_score_theory =  2*pi 

make_log  = True
n_events      = 50000
n_trees       = 50
learning_rate = 0.2 
max_depth     = 3
min_size      = 50
n_plot = 10 # Plot every tenth

weighted      = False
id_string     = "nTrees%i-sine"%(n_trees)
if weighted:id_+='-weighted'

def get_sampled_dataset( n_events ):
    features = np.array( [ [model.GetRandom(xmin, xmax)] for i in range(n_events)] )
    weights       = np.array( [1 for i in range(n_events)] ) 
    diff_weights  = np.array( [ 2*pi*sin(features[i][0]) for i in range(n_events)] )

    diff_weights*=np.random.choice([-1,1],size=len(features), p=[0.5,0.5])

    return features, weights, diff_weights

#get_dataset = get_weighted_dataset 
get_dataset = get_sampled_dataset 
