import ROOT
import numpy as np
import random

from math import sin, cos, sqrt
'''Implement the model from https://arxiv.org/pdf/1912.07628.pdf
'''

# EWSB constants
e   = 0.3028 #sqrt(4*pi*alpha)
s2w = 0.23122
sw  = sqrt(s2w)
c2w = 1-s2w
cw  = sqrt(c2w)
g   = e/sw
gZ  = g/cw
# boson masses and widths
m = {
    'W':80.379,
    'Z':91.1876,
    'H':125.1,
    }
Gamma = {
    'W':2.085,
    'Z':2.4952,
    }

sqrt_2 = sqrt(2)

N_events = 20
# qq -> ZH

# helicity of the quark
sigma_quark  = np.random.choice([+1, -1], N_events)
# boson helicity
lambda_boson = np.random.choice([+1, -1, 0], N_events)
# theta of boson in the qq restframe
cos_theta = np.random.uniform(-1,1,N_events)
sin_theta = np.sqrt( 1. - cos_theta**2 ) 
#quark
q_pdgId   = np.random.choice([1, 2],N_events)
# sqrt(s)
import pdf

E_LHC          = 13000

#abs_y_max = 3 #maximum rapidity
#sqrt_s_hat_min = m['H'] + m['Z']
#x_min          = sqrt_s_hat_min**2/E_LHC**2
#x1             = np.random.uniform( x_min, 1, N_events )
#x2             = np.random.uniform( x_min/x1, np.ones(N_events), N_events )
#shuffle        = np.random.choice([0,1], N_events).astype('bool') 
#x1[shuffle], x2[shuffle] = x2[shuffle], x1[shuffle]
#y              = 0.5*np.log(x1/x2) 
#s_hat          = E_LHC**2*x1*x2
#sqrt_s_hat     = np.sqrt(s_hat)

s_hat_min   = (m['H'] + m['Z'])**2
s_hat_max   = E_LHC**2
s_hat       = s_hat_min+(s_hat_max-s_hat_min)*np.random.uniform(0,1,N_events)
sqrt_s_hat  = np.sqrt(s_hat)
x_min       = np.sqrt( s_hat/s_hat_max )
abs_y_max   = - np.log(x_min)
y           = np.random.uniform(-1,1, N_events)*abs_y_max

w           = (s_hat + m['Z']**2-m['H']**2)/(2*np.sqrt(s_hat))
k           = np.sqrt( w**2-m['Z']**2 )

# parameters
h1Z         = 0.
h2Z         = 0.
h3Z         = 0.
h4Ztilde    = 0.
h1gamma     = 0.
h2gamma     = 0.
h3gamma     = 0.
h4gamma     = 0.

# Qq
Qq = np.zeros(N_events)
Qq[q_pdgId==1] = -1./3. #d
Qq[q_pdgId==2] =  2./3. #u
# T3q
T3q = np.zeros(N_events)
T3q[q_pdgId==1] = -1./2. #d
T3q[q_pdgId==2] =  1./2. #u

# gZsigma
gZsigma = np.zeros(N_events)
gZsigma[sigma_quark==+1] = gZ*(-Qq[sigma_quark==+1]*s2w)
gZsigma[sigma_quark==-1] = gZ*(T3q[sigma_quark==-1]-Qq[sigma_quark==-1]*s2w)

M_hat = np.zeros(N_events)
abs_lambda_1 = abs(lambda_boson)==1
M_hat[abs_lambda_1]  =  gZ * m['Z'] * sqrt_s_hat[abs_lambda_1] *( gZsigma[abs_lambda_1] / (s_hat[abs_lambda_1] - m['Z']**2)*(1.+h1Z+h2Z*s_hat[abs_lambda_1]/m['Z']**2+1j*lambda_boson[abs_lambda_1]*h4Ztilde*k[abs_lambda_1]*sqrt_s_hat[abs_lambda_1]/m['Z']**2) + Qq[abs_lambda_1]*e/s_hat[abs_lambda_1]*(h1gamma+h2gamma*s_hat[abs_lambda_1]/m['Z']**2+1j*lambda_boson[abs_lambda_1]*h4gamma*k[abs_lambda_1]*sqrt_s_hat[abs_lambda_1]/m['Z']**2 ) )
M_hat[~abs_lambda_1] = -gZ *  w[~abs_lambda_1] * sqrt_s_hat[~abs_lambda_1] *( gZsigma[~abs_lambda_1] / (s_hat[~abs_lambda_1] - m['Z']**2)*(1.+h1Z+h2Z*s_hat[~abs_lambda_1]/m['Z']**2+h3Z*k[~abs_lambda_1]**2*sqrt_s_hat[~abs_lambda_1]/(m['Z']**2*w[~abs_lambda_1])) + Qq[~abs_lambda_1]*e/s_hat[~abs_lambda_1]*(h1gamma+h2gamma*s_hat[~abs_lambda_1]/m['Z']**2+h3gamma*k[~abs_lambda_1]**2*sqrt_s_hat[~abs_lambda_1]/(m['Z']**2*w[~abs_lambda_1]) )) 

M = np.zeros(N_events)
M[abs_lambda_1]  = sigma_quark[abs_lambda_1]*(1+sigma_quark[abs_lambda_1]*lambda_boson[abs_lambda_1]*cos_theta[abs_lambda_1])/sqrt(2.)*M_hat[abs_lambda_1]
M[~abs_lambda_1] = sin_theta[~abs_lambda_1]*M_hat[~abs_lambda_1]


