import ROOT
import numpy as np
import random

from math import sin, cos, sqrt, pi
''' Analytic toy
'''

import ROOT
import csv
import os
import array
import random

# EFT settings, parameters, defaults
wilson_coefficients    = ['theta1']
tex                    = {"theta1":"#theta_{1}"}

default_eft_parameters = { 'Lambda':1000. }
default_eft_parameters.update( {var:0. for var in wilson_coefficients} )

first_derivatives = [('theta1',)]
second_derivatives= [('theta1','theta1')]
derivatives       = [tuple()] + first_derivatives + second_derivatives

def make_eft(**kwargs):
    result = { key:val for key, val in default_eft_parameters.iteritems() }
    for key, val in kwargs.iteritems():
        if not key in wilson_coefficients+["Lambda"]:
            raise RuntimeError ("Wilson coefficient not known.")
        else:
            result[key] = float(val)
    return result

random_eft = make_eft(**{v:random.random() for v in wilson_coefficients} )
sm         = make_eft()

feature_names =  ['x']

def getEvents(N_events_requested):
    return np.array( [2*pi*random.uniform(0,1) for i in range(N_events_requested)] ).reshape(-1,1)

def getWeights(features, eft):

    #dsigma/dx = (1+theta sin(x))**2, 2(1+theta*sin(x)) 
    sinx = np.sin(features[:,0])
    weights = { tuple():     (1+eft['theta1']*sinx)**2,
               ('theta1',):  2*(1+eft['theta1']*sinx)*sinx, 
               ('theta1','theta1'): 2*sinx**2,
    }

    tot_pos = np.sum(weights[()])

    p = 0.2
    pos = np.random.choice([0,1], len(features), p=[p,1-p]).astype('bool')
    for k in weights.keys():
        weights[k][~pos]*=-1
    tot_neg = np.sum(weights[()])
    for k in weights.keys():
        weights[k][~pos]*= (tot_pos/tot_neg)

    return weights

plot_options = {
    'x': {'binning':[100,0,2*pi],      'tex':"x",},
    }

eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},
    {'color':ROOT.kMagenta+2,   'eft':make_eft(theta1=-2),'tex':"#theta_{1} = -2"},
    {'color':ROOT.kMagenta-4,   'eft':make_eft(theta1=+2), 'tex':"#theta_{1} = +2"},
    {'color':ROOT.kBlue+2,      'eft':make_eft(theta1=-1),  'tex':"#theta_{1} = -1"},
    {'color':ROOT.kBlue-4,      'eft':make_eft(theta1=+1),  'tex':"#theta_{1} = +1"},
    {'color':ROOT.kGreen+2,     'eft':make_eft(theta1=-0.5),'tex':"#theta_{1} =-.5"},
    {'color':ROOT.kGreen-4,     'eft':make_eft(theta1=0.5), 'tex':"#theta_{1} =+.5"},
]

multi_bit_cfg = {'n_trees': 250,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 15 }