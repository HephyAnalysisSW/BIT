#!/usr/bin/env python
# Standard imports

import numpy as np
import random
import ROOT
from math import sqrt

# Definition of the model
## exponential
pT0   = 0.#25.
alpha = 1./800.
theta0 = alpha # define same variable in all models for plotting
xmin  = 0
xmax  = 420 
ymin  = 0
ymax  = 420 
texX  = "p_{T} [GeV]"
model            = ROOT.TF2("model", "1./{alpha}*exp(-{alpha}*(sqrt(x**2+y**2)-{pT0}))".format(pT0=pT0, alpha=alpha), xmin, xmax, ymin, ymax)
score_theory     = ROOT.TF2("score_theory", "1./{alpha}-(sqrt(x**2+y**2)-{pT0})".format(pT0=pT0, alpha=alpha), xmin, xmax, ymin, ymax)
min_score_theory = min( score_theory.Eval(xmin, ymin), score_theory.Eval(xmax, ymax) )
max_score_theory = max( score_theory.Eval(xmin, ymin), score_theory.Eval(xmax, ymax) ) 

make_log  = True
n_trees       = 500
learning_rate = 0.2 
max_depth     = 4
min_size      = 20
n_plot = 10 # Plot every tenth

weighted      = False
id_string     = "pT0%i-nTrees%i-exponential2D"%(pT0, n_trees)
if weighted:id_+='-weighted'

x, y = ROOT.Double(), ROOT.Double()
def rand_getter(model):
    model.GetRandom2(x,y)
    return [x.real, y.real]

def get_sampled_dataset( n_events ):
    features = np.array( [ rand_getter(model) for i in range(n_events)] )
    weights       = np.array( [1 for i in range(n_events)] ) 
    diff_weights  = np.array( [ (1./alpha - (sqrt(features[i][0]**2+features[i][1]**2)-pT0)) for i in range(n_events)] )
    return features, weights, diff_weights

def get_weighted_dataset( n_events ):
    features = np.array( [ [xmin+random.random()*(xmax-xmin), ymin+random.random()*(ymax-ymin)] for i in range(n_events)] )
    weights       = np.array( [ model.Eval(features[i][0],features[i][1]) for i in range(n_events)] ) 
    diff_weights  = np.array( [ weights[i]*(1./alpha - (sqrt(features[i][0]**2+features[i][1]**2)-pT0)) for i in range(n_events)] )
    return features, weights, diff_weights

get_dataset = get_weighted_dataset 
#get_dataset = get_sampled_dataset 
