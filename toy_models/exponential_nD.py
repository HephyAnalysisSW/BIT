#!/usr/bin/env python
# Standard imports

import numpy as np
import random
import ROOT
from math import exp, sqrt, pow

# Definition of the model
## exponential
pT0   = 25.
alpha = 1./100
xmin  = 20
xmax  = 420
texX  = "p_{T} [GeV]"
model = ROOT.TF1("model", "1./{alpha}*exp(-{alpha}*(x-{pT0}))".format(pT0=pT0, alpha=alpha), xmin, xmax)
score_theory     = ROOT.TF1("score_theory", "1./{alpha}-(x-{pT0})".format(pT0=pT0, alpha=alpha), xmin, xmax)
min_score_theory = min( score_theory.Eval(xmin), score_theory.Eval(xmax) )
max_score_theory = max( score_theory.Eval(xmin), score_theory.Eval(xmax) )

make_log  = True
n_trees       = 100
weighted      = False
id_string     = "pT0%i-nTrees%i-exponential"%(pT0, n_trees)
if weighted:id_+='-weighted'

# Converting x -> sqrt(x1^1+x2^2+...+xD^2)
def getSubstitude(val, nDim):
    subs = []
    remain = val
    remain2 = pow(val,2)
    for i in range(nDim-1):
        s2 = random.uniform(0, remain2)
        remain2 = remain2 - s2
        remain = sqrt(remain2)
        subs.append(sqrt(s2))
    subs.append(remain)
    return subs

def get_sampled_dataset( n_events, n_dim ):
    features = np.array(  [getSubstitude(model.GetRandom(xmin, xmax), n_dim) for i in range(n_events)] )
    weights       = np.array( [1 for i in range(n_events)] )
    features_list = []
    for i in range(n_events):
        feature_sum2 = 0
        for j in range(n_dim):
            feature_sum2 += pow(features[i][j],2)
        feature_term = 1./alpha - (sqrt(feature_sum2)-pT0)
        features_list.append(feature_term)
    diff_weights  = np.array( features_list )
    return features, weights, diff_weights

get_dataset = get_sampled_dataset
