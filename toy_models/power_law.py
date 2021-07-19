#!/usr/bin/env python
# Standard imports

import numpy as np
import ROOT

from math import log

# power law
# (pT/pT0)^(theta) pT^(-alpha) -> the score is Log[pT/pT0]
xmin  = 20
xmax  = 420 
alpha = 3
theta0 = alpha # define same variable in all models for plotting
pT0   = 100
texX  = "p_{T} [GeV]"
model = ROOT.TF1("model", "x^(-{alpha})".format(alpha=alpha), xmin,xmax)
score_theory = ROOT.TF1("score_theory", "1./({alpha}-1)-log(x/{pT0})".format(alpha=alpha, pT0=pT0), xmin, xmax)
min_score_theory = min( score_theory.Eval(xmin), score_theory.Eval(xmax) )
max_score_theory = max( score_theory.Eval(xmin), score_theory.Eval(xmax) ) 
make_log  = True
n_events      = 100000
n_trees       = 150
learning_rate = .5 
max_depth     = 2
min_size      = 50
n_plot        = 10 
id_string   = "power-law-OT-nTrees%i-LR%3.2f-pT%iTo%i"%(n_trees, learning_rate, xmin, xmax)
def get_dataset( n_events ):
    features = np.array( [ [model.GetRandom(xmin, xmax)] for i in range(n_events)] )
    weights       = np.array( [1 for i in range(n_events)] ) 
    diff_weights  = np.array( [ 1./(alpha-1.) - log(features[i][0]/pT0) for i in range(n_events)] )
    return features, weights, diff_weights

