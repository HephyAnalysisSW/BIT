import ROOT
import numpy as np

from math import sin, cos, sqrt
'''Implement the model from https://arxiv.org/pdf/1912.07628.pdf
'''

# Eq. 3.6
f = {
    0: lambda Theta, theta, phi: sin(Theta)**2*sin(theta)**2,
    1: lambda Theta, theta, phi: cos(Theta)*cos(theta),
    2: lambda Theta, theta, phi: (1+cos(Theta)**2)*(1+cos(theta)**2),
    3: lambda Theta, theta, phi: cos(phi)*sin(Theta)*sin(theta),
    4: lambda Theta, theta, phi: cos(phi)*sin(Theta)*sin(theta)*cos(Theta)*cos(theta),
    5: lambda Theta, theta, phi: sin(phi)*sin(Theta)*sin(theta),
    6: lambda Theta, theta, phi: sin(phi)*sin(Theta)*sin(theta)*cos(Theta)*cos(theta),
    7: lambda Theta, theta, phi: cos(2*phi)*sin(Theta)**2*sin(theta)**2,
    8: lambda Theta, theta, phi: sin(2*phi)*sin(Theta)**2*sin(theta)**2,
}

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

# 1st line p9
G_V = {
    'W': {
        'lL':g**2/sqrt(2),
        'lR':0.,
        },
    'Z': {
        'lL':g*g_l_V['Z']['lL']/cw,
        'lR':g*g_l_V['Z']['lR']/cw,
        },
    }
 
# Eq. 3.7
eps_RL  = {V: (g_l_V[V]['lR']**2 -g_l_V[V]['lL']**2)/(g_l_V[V]['lR']**2+g_l_V[V]['lL']**2) for V in ['W','Z'] }
# Caption Tab. 2
mathcal_G_V   = {V:{l:g*g_l_V[V][l]*sqrt(g_l_V[V]['lL']**2+g_l_V[V]['lR']**2)/(cw*Gamma[V]) for l in ['lL', 'lR']} for V in ['Z', 'W']}

# Eq. 2.4
def kappa_gamma_gamma(eft):
    return 2*(eft['v']**2/eft['Lambda'])**2*( s2w*eft['cHW'] + c2w*eft['cHB'] - sw*cw*eft['cHWB'] )
def kappa_Z_gamma(eft):
    return (eft['v']**2/eft['Lambda'])**2*(2*cw*sw*(eft['cHW']-eft['cHB'])+(s2w-c2w)*eft['cHWB'])
def kappa_tilde_Z_gamma(eft):
    return (eft['v']**2/eft['Lambda'])**2*(2*cw*sw*(eft['cHW_tilde']-eft['cHB_tilde'])+(s2w-c2w)*eft['cHWB_tilde'])
def kappa_WW(eft):
    return 2*(eft['v']**2/eft['Lambda'])**2*eft['cHW']
def kappa_tilde_WW(eft):
    return 2*(eft['v']**2/eft['Lambda'])**2*eft['cHW_tilde']
def kappa_ZZ(eft):
    return 2*(eft['v']**2/eft['Lambda'])**2*(c2w*eft['cHW']+s2w*eft['cHB']+sw*cw*eft['cHWB']
def kappa_tilde_ZZ(eft):
    return 2*(eft['v']**2/eft['Lambda'])**2*(c2w*eft['cHW_tilde']+s2w*eft['cHB_tilde']+sw*cw*eft['cHWB_tilde']

def delta_mZ2_over_MZ2( eft ):
    return (eft['v']**2/eft['Lambda'])**2*(2*sw/cw*eft['cWB']+0.5*eft['cHD'])    

delta_g_Z = {f: g*Y[f]*sw/c2w*(eft['v']**2/eft['Lambda'])**2*eft['cWB']-g/cw*(eft['v']**2/eft['Lambda'])**2*(abs(T3[f])*eft['c1HF']-T3[f]*eft['c3HF']+(0.5-abs(T3[f])*eft['cH'+f[0]])+delta_mZ2_over_mZ2(eft)*g/(2.*cw*s2w)*(T3[f]*c2w+Y[f]*s2w))  for f in ['uL', 'dL']}

def g_h_W_Q(eft)
    return sqrt(2)*g*(eft['v']**2/eft['Lambda'])**2*eft['c3HQ']

delta_g_hat_h_VV = { 'W': lambda eft: return (eft['v']/eft['Lambda'])**2*(eft['cHBox']-0.25*eft['cHD']), #Eq. 2.1
                     # Eq. 2.5
                     'Z': lambda eft: return delta_g_hat_h_VV['W'](eft) - (kappa_WW(eft)-kappa_gamma_gamma(eft)-kappa_Z_gamma(eft)*cw/sw))*s2w/c2w + (sqrt(2)*cw*(delta_g_Z_f['uL']-delta_g_Z_f['dL'])-g_h_W_Q(eft))*s2w/(sqrt(2)*g*c2w),

# 3.2
def kappa_hat_VV(eft) :
    return kappa_WW(eft)
def kappa_hat_ZZ(eft,l):
    return kappa_ZZ(eft)+Q[l]*e/g_l_V['V'][l]*kappa_Z_gamma(eft)
def kappa_hat_tilde_ZZ(eft,l):
    return kappa_tilde_ZZ(eft)+Q[l]*e/g_l_V['V'][l]*kappa_tilde_Z_gamma(eft)

class VH:
    def __init__( self,  V ):
        self.V = V

    gamma = sqrt(s_hat)/(2*m[V])

    0.25 *mathcal_G_V[V][l]**2                          *( 1. + 2.*delta_g_hat_h_VV[V](eft)+4.*kappa_hat_VV(eft)+2.*delta_g_Z_f+(g_h_V_f/g_V_f)*(-1+4.*gamma^2)
    0.5  *mathcal_G_V[V][l]**2*sigma*eps_RL*gamma**(-2) *(1+4*(g_h_V_f/g_V_f+kappa_hat_VV)*gamma**2)
    0.125*mathcal_G_V[V][l]**2*gamma**(-2)              *(1+4*(g_h_V_f/g_V_f+kappa_hat_VV)*gamma**2)
    -0.5 *mathcal_G_V[V][l]**2*sigma*eps_RL/gamma       *(1+2*(2*g_h_V_f/g_V_f+kappa_hat_VV)*gamma**2)
    -0.5 *mathcal_G_V[V][l]**2/gamma                    *(1+2*(2*g_h_V_f/g_V_f+kappa_hat_VV)*gamma**2)
    -     mathcal_G_V[V][l]**2*sigma*eps_RL*kappa_tilde_hat_VV*gamma
    -     mathcal_G_V[V][l]**2*kappa_tilde_hat_VV*gamma
    0.125*mathcal_G_V[V][l]**2*(1+4*(g_h_V_f/g_V_f+kappa_hat_VV)*gamma**2)
    0.5*  mathcal_G_V[V][l]**2*kappa_tilde_hat_VV

    def amp_sq(self, s_hat, Theta, theta, phi):
        '''Eq. 3.5'''
        return np.array([self.a[i](s_hat)*f[i](Theta, theta, phi)] for i in range(9)).sum()
        
