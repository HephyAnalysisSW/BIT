import ROOT
import numpy as np
import random

from math import sin, cos, sqrt, pi
'''Implement the model from https://arxiv.org/pdf/2007.10356.pdf'''


# EWSB constants
e   = 0.3028 #sqrt(4*pi*alpha)
s2w = 0.23122
sw  = sqrt(s2w)
c2w = 1-s2w
cw  = sqrt(c2w)
g   = e/sw

# boson masses and widths
m = {
    'W':80.379,
    'Z':91.1876,
    }
Gamma = {
    'W':2.085,
    'Z':2.4952,
    }

# SM quantum numbers
T3 = {
    'lL': -0.5,
    'lR':  0.,
    'uL': 0.5,
    'dL': -0.5,
    }
Q = {
    'lL': -1.,
    'lR': -1.,
    }
Y  = {
    'uL':1/3,
    'dL':1/3,
    'lL':-1.,
    'lR':-2.,
}

# V-lepton couplings
gW = g/sqrt(2)
gL = -g(1-2*s2w)/(2.*cw)
gR = g*s2w/cw

# EFT settings, parameters, defaults
wilson_coefficients    = ['c3PQ', 'cW']
observables = ["s", "Theta", 'thetaW', 'phiW', 'thetaZ', 'phiZ', 'pTZW' ]
default_eft_parameters = { }
default_eft_parameters.update( {var:0. for var in wilson_coefficients} )

def make_eft(**kwargs):
    result = { key:val for key, val in default_eft_parameters.iteritems() }
    for key, val in kwargs.iteritems():
        if not key in wilson_coefficients:
            raise RuntimeError ("Wilson coefficient not known.")
        else:
            result[key] = float(val)
    return result 

random_eft = make_eft(**{v:random.random() for v in wilson_coefficients} )

# Eq. 17 and derivatives
def M(h1, h1, s, Theta, eft, der=None):
    if h1==h2==0:
        if der is None:
            return -g**2*np.sin(Theta)/(2*sqrt(2))-sqrt(2)*eft['c3PQ']*s*np.sin(Theta)
        elif der=='c3PQ':
            return -g**2*np.sin(Theta)/(2*sqrt(2))-sqrt(2)*s*np.sin(Theta)
        else:
            return 0.
    elif h1==h2==1 or h1==h2==-1:
        if der is None:
            return 3*g*cw*eft['cW']*s*np.sin(Theta)/sqrt(2)
        elif der=='cW':
            return 3*g*cw*s*np.sin(Theta)/sqrt(2)
        else:
            return 0.
    elif h1=-1 and h2==1:
        if der is not None: return 0.
        return -g**2*(s2w-3*c2w*np.cos(Theta))/(3*sqrt(2)*cw)*np.cot(0.5*Theta)
    elif h1=1 and h2==-1:
        if der is not None: return 0.
        return  g**2*(s2w-3*c2w*np.cos(Theta))/(3*sqrt(2)*cw)*np.tan(0.5*Theta)
    else:
        return 0.

prefactor = m['W']*m['Z']/(6.*Gamma['W']*Gamma['Z'])

class WZ:
    def __init__( self, eft ):
        self.eft = eft
        self.feature_names = ['Theta', 'phiW', 'phiZ', 'thetaW', 'thetaZ'] 

    def getEvents( nEvents ): 
        Theta   = np.arccos( -1+2*np.random.random(nEvents) )
        phiW    = 2*pi*np.random.random(nEvents)
        phiZ    = 2*pi*np.random.random(nEvents)
        thetaW  = np.arccos( -1+2*np.random.random(nEvents) )
        thetaZ  = np.arccos( -1+2*np.random.random(nEvents) )
        return np.transpose(np.array( [Theta, phiW, phiZ,  thetaW, thetaZ]))

    def getWeights( features ): 

        Theta, phiW, phiZ, thetaW, thetaZ = np.transpose(features)
        for hZ in [-1,0,1]:
            for hZp in [-1,0,1]:
                for hW in [-1,0,1]:
                    for hWp in [-1,0,1]:
                        cos(0.5*(hW-hWp)*(pi-2*phiW))*M(hZ, hW, s, Theta, eft)*gR**2*M(hZp, hWp,  s, Theta, eft)*d[hW](thetaW)*d[hWp](thetaW)*( \
                            cos(0.5*pi*(hW-hWp)+(hZ-hZp)*(pi+phiZ)*gR**2*d[hZ](pi-thetaZ)*d[hZp](pi-thetaZ))
                           +cos(0.5*pi*(hW-hWp)+(hZ-hZp)*phiZ)*gL**2*d[hZ](thetaZ)*d[hZp](thetaZ))
