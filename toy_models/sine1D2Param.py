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

# EFT settings, parameters, defaults
wilson_coefficients    = ['theta1', 'theta2']
tex                    = {"theta1":"#theta_{1}", "theta2":"#theta_{2}"}

default_eft_parameters = { 'Lambda':1000. }
default_eft_parameters.update( {var:0. for var in wilson_coefficients} )

first_derivatives = [('theta1',), ('theta2',)]
second_derivatives= [('theta1','theta1'), ('theta1','theta2'), ('theta2', 'theta2')]
derivatives       = [tuple()] + first_derivatives + second_derivatives

def make_eft(**kwargs):
    result = { key:val for key, val in default_eft_parameters.items() }
    for key, val in kwargs.items():
        if not key in wilson_coefficients+["Lambda"]:
            raise RuntimeError ("Wilson coefficient not known.")
        else:
            result[key] = float(val)
    return result

random_eft = make_eft(**{v:random.random() for v in wilson_coefficients} )
sm         = make_eft()

feature_names =  ['x']

def getEvents(N_events_requested):

    #x = 0.5*np.ones(N_events_requested) 
    x = 2*pi*np.random.rand(N_events_requested) 

    return np.transpose(np.array( [x] ))

def getWeights(features, eft=default_eft_parameters):

    #dsigma/dx = (1+theta1*sin(x)+theta2*cos(x))**2 = 1 + 2 theta1 sin(x) + 2 theta 2 cos(y) + theta1**2 sin(x)**2 + theta2**2 cos(x)**2 + 2 theta1 theta2 sin(x) cos(x) 

    weights = { tuple():     np.ones(len(features)),
               ('theta1',):         2*np.sin(features[:,0]), 
               ('theta1','theta1'): 2*np.sin(features[:,0])**2,
               ('theta2',):         2*np.cos(features[:,0]), 
               ('theta2','theta2'): 2*np.cos(features[:,0])**2,
               ('theta1','theta2'): 2*np.sin(features[:,0])*np.cos(features[:,0]),
    }

    return weights

plot_options = {
    'x': {'binning':[50,0,2*pi],      'tex':"x",},
    }

eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},
    {'color':ROOT.kBlue+2,      'eft':make_eft(theta1=-1),  'tex':"#theta_{1} = -1"},
    {'color':ROOT.kBlue-4,      'eft':make_eft(theta1=+1),  'tex':"#theta_{1} = +1"},
    {'color':ROOT.kMagenta+2,   'eft':make_eft(theta2=-1),  'tex':"#theta_{2} = -1"},
    {'color':ROOT.kMagenta-4,   'eft':make_eft(theta2=+1),  'tex':"#theta_{2} = +1"},
#    {'color':ROOT.kGreen+2,     'eft':make_eft(theta1=-0.5),'tex':"#theta_{1} =-.5"},
#    {'color':ROOT.kGreen-4,     'eft':make_eft(theta1=0.5), 'tex':"#theta_{1} =+.5"},
]

multi_bit_cfg = {'n_trees': 250,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 15 }
