import ROOT
import numpy as np
import random

from math import sin, cos, sqrt
'''Implement the model from https://arxiv.org/pdf/1912.07628.pdf
'''

# Eq. 3.6
def f_i(Theta, theta, phi):
    return np.array([\
     sin(Theta)**2*sin(theta)**2,
     cos(Theta)*cos(theta),
     (1+cos(Theta)**2)*(1+cos(theta)**2),
     cos(phi)*sin(Theta)*sin(theta),
     cos(phi)*sin(Theta)*sin(theta)*cos(Theta)*cos(theta),
     sin(phi)*sin(Theta)*sin(theta),
     sin(phi)*sin(Theta)*sin(theta)*cos(Theta)*cos(theta),
     cos(2*phi)*sin(Theta)**2*sin(theta)**2,
     sin(2*phi)*sin(Theta)**2*sin(theta)**2,
    ])

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
g_l_V  = {
    'W': { 'lL':g/sqrt(2),
           'lR':0.,
         },
    'Z': {'lL':g*(T3['lL']-Q['lL']*s2w)/cw,
          'lR':g*(T3['lR']-Q['lR']*s2w)/cw,
         },
}

## 1st line p9
#G_V = {
#    'W': {
#        'lL':g**2/sqrt(2),
#        'lR':0.,
#        },
#    'Z': {
#        'lL':g*g_l_V['Z']['lL']/cw,
#        'lR':g*g_l_V['Z']['lR']/cw,
#        },
#    }

# EFT settings, parameters, defaults
wilson_coefficients    = ['cHW', 'cHB', 'cHWB', 'cHW_tilde', 'cHB_tilde', 'cHWB_tilde', 'cWB', 'cHD', 'cHBox']
default_eft_parameters = { 'Lambda':1000., 'v':246. }
default_eft_parameters.update( {var:0. for var in wilson_coefficients} )

def make_eft(**kwargs):
    result = default_eft_parameters
    for key, val in kwargs.iteritems():
        if not key in wilson_coefficients:
            raise RuntimeError ("Wilson coefficient not known.")
        else:
            result[key] = float(val)
    return result 

random_eft = make_eft(**{v:random.random() for v in wilson_coefficients} )

# Eq. 3.7
eps_RL  = {V: (g_l_V[V]['lR']**2 -g_l_V[V]['lL']**2)/(g_l_V[V]['lR']**2+g_l_V[V]['lL']**2) for V in ['W','Z'] }
# Caption Tab. 2
mathcal_G_V   = {V:{l:g*g_l_V[V][l]*sqrt(g_l_V[V]['lL']**2+g_l_V[V]['lR']**2)/(cw*Gamma[V]) for l in ['lL', 'lR']} for V in ['Z', 'W']}

# Eq. 2.4
#def kappa_gamma_gamma(eft):
#    return 2*(eft['v']**2/eft['Lambda'])**2*( s2w*eft['cHW'] + c2w*eft['cHB'] - sw*cw*eft['cHWB'] )
def kappa_Z_gamma(eft):
    return (eft['v']/eft['Lambda'])**2*(2*cw*sw*(eft['cHW']-eft['cHB'])+(s2w-c2w)*eft['cHWB'])
def kappa_tilde_Z_gamma(eft):
    return (eft['v']/eft['Lambda'])**2*(2*cw*sw*(eft['cHW_tilde']-eft['cHB_tilde'])+(s2w-c2w)*eft['cHWB_tilde'])
def kappa_WW(eft):
    return 2*(eft['v']/eft['Lambda'])**2*eft['cHW']
def kappa_tilde_WW(eft):
    return 2*(eft['v']/eft['Lambda'])**2*eft['cHW_tilde']
def kappa_ZZ(eft):
    return 2*(eft['v']/eft['Lambda'])**2*(c2w*eft['cHW']+s2w*eft['cHB']+sw*cw*eft['cHWB'])
def kappa_tilde_ZZ(eft):
    return 2*(eft['v']/eft['Lambda'])**2*(c2w*eft['cHW_tilde']+s2w*eft['cHB_tilde']+sw*cw*eft['cHWB_tilde'])

def delta_mZ2_over_mZ2( eft ):
    return (eft['v']/eft['Lambda'])**2*(2*sw/cw*eft['cWB']+0.5*eft['cHD'])

delta_g_Z = {l: lambda eft, l=l: g*Y[l]*sw/c2w*(eft['v']/eft['Lambda'])**2*eft['cWB'] + delta_mZ2_over_mZ2(eft)*g/(2.*cw*s2w)*(T3[l]*c2w+Y[l]*s2w) for l in ['lL', 'lR'] }
#delta_g_W = {l: lambda eft, l=l: delta_mZ2_over_mZ2(eft)*sqrt(2)*g*c2w/(4*s2w) for l in ['lL', 'lR'] }
#g/cw*(eft['v']/eft['Lambda'])**2*(abs(T3[f])*eft['c1HF']-T3[f]*eft['c3HF']+(0.5-abs(T3[f])*eft['cH'+f[0]])+delta_mZ2_over_mZ2(eft)*g/(2.*cw*s2w)*(T3[f]*c2w+Y[f]*s2w))  for f in ['uL', 'dL']}

#def g_h_W_Q(eft)
#    return sqrt(2)*g*(eft['v']/eft['Lambda'])**2*eft['c3HQ']

delta_g_hat_h_VV =  {'W': lambda eft: (eft['v']/eft['Lambda'])**2*(eft['cHBox']-0.25*eft['cHD']) } # Eq. 2.1
delta_g_hat_h_VV    ['Z'] =  lambda eft: delta_g_hat_h_VV['W'](eft) - s2w/c2w*kappa_WW(eft)        # Eq. 2.6 (LEP simplification)

# Eq 3.2
kappa_hat_VV = {'W': { 'lL': kappa_WW, # Eq. 3.2
                       'lR': kappa_WW},# Eq. 3.2
                'Z': { 'lL': lambda eft: kappa_ZZ(eft)+Q['lL']*e/g_l_V['Z']['lL']*kappa_Z_gamma(eft),
                       'lR': lambda eft: kappa_ZZ(eft)+Q['lR']*e/g_l_V['Z']['lR']*kappa_Z_gamma(eft)
                     }}
kappa_tilde_hat_VV = { 
                'W': { 'lL': kappa_tilde_WW,    # assuming the hat is superflous for kappa-tilde WW !!!  
                       'lR': kappa_tilde_WW},   # The 'L/R' implementation of kappa_tilde_WW is just so that kappa_tilde_hat_VV can be a nested dictionary for VV=ZZ/WW (kappa_tilde_hat_VV['W']['lR']!=0)
                'Z': { 'lL': lambda eft: kappa_ZZ(eft)+Q['lL']*e/g_l_V['Z']['lL']*kappa_Z_gamma(eft),
                       'lR': lambda eft: kappa_tilde_ZZ(eft)+Q['lR']*e/g_l_V['Z']['lR']*kappa_tilde_Z_gamma(eft)
                     }}

def a_i(s_hat, V, l, sigma, eft):

    gamma = sqrt(s_hat)/(2*m[V]) # Caption Tab. 2

    return np.array([\
      0.25 *mathcal_G_V[V][l]**2                          *( 1. + 2.*delta_g_hat_h_VV[V](eft)+4.*kappa_hat_VV[V][l](eft)+2.*delta_g_Z[l](eft)),
      0.5  *mathcal_G_V[V][l]**2*sigma*eps_RL[V]*gamma**(-2) *(1+4*(kappa_hat_VV[V][l](eft))*gamma**2),
      0.125*mathcal_G_V[V][l]**2*gamma**(-2)              *(1+4*(kappa_hat_VV[V][l](eft))*gamma**2),
      -0.5 *mathcal_G_V[V][l]**2*sigma*eps_RL[V]/gamma       *(1+2*(kappa_hat_VV[V][l](eft))*gamma**2),
      -0.5 *mathcal_G_V[V][l]**2/gamma                    *(1+2*(kappa_hat_VV[V][l](eft))*gamma**2),
      -     mathcal_G_V[V][l]**2*sigma*eps_RL[V]*kappa_tilde_hat_VV[V][l](eft)*gamma,
      -     mathcal_G_V[V][l]**2*kappa_tilde_hat_VV[V][l](eft)*gamma,
      0.125*mathcal_G_V[V][l]**2*(1+4*(kappa_hat_VV[V][l](eft))*gamma**2),
      0.5*  mathcal_G_V[V][l]**2*kappa_tilde_hat_VV[V][l](eft),
    ])

class VH:
    def __init__( self ):

    def amp_sq(self, s_hat, V, l, sigma, Theta, theta, phi, eft):
        '''Eq. 3.5'''
        return np.dot( a_i(s_hat, V, l, sigma, eft), f_i(Theta, theta, phi))
