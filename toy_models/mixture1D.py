#!/usr/bin/env python
# Standard imports

import numpy as np
import random
import ROOT
from math import exp

# Definition of the model
## exponential
x0    = 20.
alpha1 = 2./100
alpha2 = 1./100
theta0 = 0 
xmin  = 20
xmax  = 220 
texX  = "x"
model = ROOT.TF1("model", "(2*(exp(-((x-{x0})*{alpha1})) + {theta}/exp((x-{x0})*{alpha2}))**2)/(1./{alpha1} + {theta}*(4./({alpha1} + {alpha2}) + {theta}/{alpha2}))".format(x0=x0,alpha1=alpha1,alpha2=alpha2, theta=theta0), xmin, xmax)
score_theory     = ROOT.TF1("score_theory", "((1/alpha1 + theta*(4/(alpha1 + alpha2) + theta/alpha2))*((-2*(exp(-((x - x0)*alpha1)) + theta/exp((x - x0)*alpha2))**2*(4/(alpha1 + alpha2) + (2*theta)/alpha2))/ (1/alpha1 + theta*(4/(alpha1 + alpha2) + theta/alpha2))**2 + (4*(exp(-((x - x0)*alpha1)) + theta/exp((x - x0)*alpha2)))/(exp((x - x0)*alpha2)*(1/alpha1 + theta*(4/(alpha1 + alpha2) + theta/alpha2)))))/(2.*(exp(-((x - x0)*alpha1)) + theta/exp((x - x0)*alpha2))**2)".format(x0=x0,alpha1=alpha1,alpha2=alpha2, theta=theta0), xmin, xmax)
min_score_theory = min( score_theory.Eval(xmin), score_theory.Eval(xmax) )
max_score_theory = max( score_theory.Eval(xmin), score_theory.Eval(xmax) ) 

make_log  = True
n_events      = 100000
n_trees       = 100
learning_rate = 0.2 
max_depth     = 2
min_size      = 50
n_plot = 10 # Plot every tenth

weighted      = False
id_string     = "pT0%i-nTrees%i-exponential"%(pT0, n_trees)
if weighted:id_+='-weighted'

def get_sampled_dataset( n_events ):
    features = np.array( [ [model.GetRandom(xmin, xmax)] for i in range(n_events)] )
    weights       = np.array( [1 for i in range(n_events)] ) 
    diff_weights  = np.array( [ ( ((1./alpha1 + theta0*(4./(alpha1 + alpha2) + theta0/alpha2))*((-2.*(exp(-((x - x0)*alpha1)) + theta0/exp((x - x0)*alpha2))**2*(4./(alpha1 + alpha2) + (2*theta0)/alpha2))/(1./alpha1 + theta0*(4./(alpha1 + alpha2) + theta0/alpha2))**2 + (4.*(exp(-((x - x0)*alpha1)) + theta0/exp((x - x0)*alpha2)))/(exp((x - x0)*alpha2)*(1/alpha1 + theta0*(4./(alpha1 + alpha2) + theta0/alpha2)))))/ (2.*(exp(-((x - x0)*alpha1)) + theta0/exp((x - x0)*alpha2))**2)) for i in range(n_events)] )
    return features, weights, diff_weights
#def get_weighted_dataset( n_events ):
#    features = np.array( [ [xmin+random.random()*(xmax-xmin)] for i in range(n_events)] )
#    weights       = np.array( [ model.Eval(features[i][0]) for i in range(n_events)] ) 
#    diff_weights  = np.array( [ weights[i]*(1./theta0 - (features[i][0]-pT0)) for i in range(n_events)] )
#    return features, weights, diff_weights

#get_dataset = get_weighted_dataset 
get_dataset = get_sampled_dataset 
