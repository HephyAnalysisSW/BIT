# Standard imports

import numpy as np
import ROOT

id_string = "flat"
xmin      = 0
pT0       = 200 
xmax      = 2*pT0
model     = ROOT.TF1("model", "1", xmin, xmax)
max_depth = 2
min_size  = 50
alpha     = 1.
texX  = "p_{T} [GeV]"
score_theory = ROOT.TF1("score_theory", "-1./(2+{alpha})+1./(1+{alpha})*(x>{pT0})".format(alpha=alpha, pT0=pT0), xmin, xmax)
min_score_theory = min( score_theory.Eval(xmin), score_theory.Eval(xmax) )
max_score_theory = max( score_theory.Eval(xmin), score_theory.Eval(xmax) ) 
make_log  = False
n_events      = 10000
n_trees       = 20
n_plot        = 5
learning_rate = 0.1
id_   = "piece-wise-nTrees%i-LR%3.2f-pT%iTo%i"%(n_trees, learning_rate, xmin, xmax)
def get_dataset( n_events ):
    ''' Produces data set according to theory model'''
    features = np.array( [ [model.GetRandom(xmin, xmax)] for i in range(n_events)] )
    weights       = np.array( [ 1. for i in range(n_events)] ) 
    diff_weights  = np.array( [ -1./(alpha+2)+ (1./(alpha+1.) if features[i]>pT0 else 0) for i in range(n_events)] )
    return features, weights, diff_weights
