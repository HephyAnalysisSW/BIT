import ROOT
import numpy as np
import random

from math import sin, cos, sqrt, pi
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

N_events = 10
# qq -> ZH

# helicity of the quark
sigma_quark  = np.random.choice([+1, -1], N_events)
# boson helicity
lambda_boson = np.random.choice([+1, -1, 0], N_events)
# theta of boson in the qq restframe
cos_theta = np.random.uniform(-1,1,N_events)
sin_theta = np.sqrt( 1. - cos_theta**2 ) 

phi_hat = pi*np.random.uniform(-1,1,N_events)
cos_theta_hat = np.random.uniform(-1,1,N_events)
sin_theta_hat = np.sqrt( 1. - cos_theta**2 ) 

#quark
q_pdgId   = np.random.choice([1, 2],N_events)
# sqrt(s)
import pdf

E_LHC          = 13000

# Qq
Qq  = {1:-1./3., 2: 2./3.}
T3q = {1:-.5,    2:.5}

#pp -> ZH

# kinematics
s_hat_min   = (m['H'] + m['Z'])**2
s_hat_max   = E_LHC**2
s_hat       = s_hat_min+(s_hat_max-s_hat_min)*np.random.uniform(0,0.005,N_events)
sqrt_s_hat  = np.sqrt(s_hat)
x_min       = np.sqrt( s_hat/s_hat_max )
abs_y_max   = - np.log(x_min)
y           = np.random.uniform(-1,1, N_events)*abs_y_max

w_Z           = (s_hat + m['Z']**2-m['H']**2)/(2*np.sqrt(s_hat))
k_Z           = np.sqrt( w_Z**2-m['Z']**2 )

x1          = sqrt_s_hat/E_LHC*np.exp(y)
x2          = sqrt_s_hat/E_LHC*np.exp(-y)

# parameters
h1Z         = 0.
h2Z         = 0.
h3Z         = 0.
h4Ztilde    = 0.
h1gamma     = 0.
h2gamma     = 0.
h3gamma     = 0.
h4gamma     = 0.

Cf  = 1.
#gZsigma[sigma_quark][pdg_quark]
gZsigma = {1:{1: gZ*(-Qq[1]*s2w), 2: gZ*(-Qq[2]*s2w)}, -1:{1:gZ*(T3q[1]-Qq[1]*s2w), 2:gZ*(T3q[2]-Qq[2]*s2w)}}

gZtaull = {1:gZ*s2w, -1:gZ*(-0.5+s2w) }

first_derivatives = [
        ('h1Z',), ('h2Z',), ('h3Z',), ('h4Ztilde',), 
        ('h1gamma',), ('h2gamma',), ('h3gamma',), ('h4gammatilde',), 
]
second_derivatives= [ ('h1Z','h1Z'), ('h2Z','h2Z'), ('h3Z', 'h3Z'), ('h4Ztilde', 'h4Ztilde'),\
                      ('h1gamma','h1gamma'), ('h2gamma','h2gamma'), ('h3gamma', 'h3gamma'), ('h4gammatilde', 'h4gammatilde'),\
                      ('h1Z','h2Z'),  ('h1Z','h3Z'), ('h1Z', 'h4Ztilde'), ('h1Z', 'h1gamma'), ('h1Z', 'h2gamma'), ('h1Z', 'h3gamma'), ('h1Z', 'h4gammatilde'),\
                      ('h2Z', 'h3Z'), ('h2Z', 'h4Ztilde'), ('h2Z', 'h1gamma'), ('h2Z', 'h2gamma'), ('h2Z', 'h3gamma'), ('h2Z', 'h4gammatilde'),\
                      ('h3Z', 'h4Ztilde'), ('h3Z', 'h1gamma'), ('h3Z', 'h2gamma'), ('h3Z', 'h3gamma'), ('h3Z', 'h4gammatilde'),\
                      ('h4Ztilde', 'h1gamma'), ('h4Ztilde', 'h2gamma'), ('h4Ztilde', 'h3gamma'), ('h4Ztilde', 'h4gammatilde'),\
                      ('h1gamma', 'h2gamma'), ('h1gamma', 'h3gamma'), ('h1gamma', 'h4gammatilde'),\
                      ('h2gamma', 'h3gamma'), ('h2gamma', 'h4gammatilde'),\
                      ('h3gamma', 'h4gammatilde'),\
]

derivatives   = [ tuple() ] + first_derivatives + second_derivatives

#constZH = 12288*pi**3*Gamma['Z']*E_LHC**2
constZH = 1 
dsigmaZH  = np.zeros(N_events).astype('complex128')
for pdg_quark in [1, 2]:
    qx1     = np.array( [ pdf.pdf( x,  pdg_quark ) for x in x1 ] ) 
    qbarx2  = np.array( [ pdf.pdf( x, -pdg_quark ) for x in x2 ] ) 

    qbarx1  = np.array( [ pdf.pdf( x, -pdg_quark ) for x in x1 ] ) 
    qx2     = np.array( [ pdf.pdf( x,  pdg_quark ) for x in x2 ] ) 
    for sigma_quark in [+1, -1]:
        dtau = {}
        M_lambda_sigma_qbarq = {}
        M_lambda_sigma_qqbar = {}
        for lambda_boson in [+1, -1, 0]:
            if abs(lambda_boson)==1:
                Mhat = gZ * m['Z'] * sqrt_s_hat\
                    * ( gZsigma[sigma_quark][pdg_quark] / (s_hat - m['Z']**2)*(1.+h1Z+h2Z*s_hat/m['Z']**2+1j*lambda_boson*h4Ztilde*k_Z*sqrt_s_hat/m['Z']**2)
                    +   Qq[pdg_quark]*e/s_hat*(h1gamma+h2gamma*s_hat/m['Z']**2+1j*lambda_boson*h4gammatilde*k_Z*sqrt_s_hat/m['Z']**2 ))
                M_lambda_sigma_qqbar[lambda_boson] =  sigma_quark*(1+sigma_quark*lambda_boson*cos_theta)/sqrt(2.)*Mhat 
                M_lambda_sigma_qbarq[lambda_boson] = -sigma_quark*(1-sigma_quark*lambda_boson*cos_theta)/sqrt(2.)*Mhat 
            else:
                M_lambda_sigma_qqbar[lambda_boson] = sin_theta*(-gZ) * w_Z * sqrt_s_hat\
                    * ( gZsigma[sigma_quark][pdg_quark] / (s_hat - m['Z']**2)*(1.+h1Z+h2Z*s_hat/m['Z']**2+h3Z*k_Z**2*sqrt_s_hat/(m['Z']**2*w_Z)) 
                    + Qq[pdg_quark]*e/s_hat*(h1gamma+h2gamma*s_hat/m['Z']**2+h3gamma*k_Z**2*sqrt_s_hat/(m['Z']**2*w_Z) ))
                M_lambda_sigma_qbarq[lambda_boson] = M_lambda_sigma_qqbar[lambda_boson]

            dtau[lambda_boson] = {}
            for tau in [+1, -1]:
                if abs(tau)==1:
                    dtau[lambda_boson][tau] = tau*(1+lambda_boson*tau*cos_theta_hat)/sqrt(2.)*np.exp(1j*lambda_boson*phi_hat) 
                else:
                    dtau[lambda_boson][tau] = sin_theta_hat 

        for lam1 in [+1, -1, 0]:
            for lam2 in [+1, -1, 0]:
                for tau in [+1, -1]:
                    dsigmaZH += m['Z']*k_Z/(constZH*s_hat**1.5)*2*gZtaull[tau]**2*Cf*(
                        qx1*qbarx2*np.conjugate(dtau[lam1][tau])*np.conjugate(M_lambda_sigma_qqbar[lam1])*M_lambda_sigma_qqbar[lam2]*dtau[lam2][tau]
                        + qbarx1*qx2*np.conjugate(dtau[lam1][tau])*np.conjugate(M_lambda_sigma_qbarq[lam1])*M_lambda_sigma_qbarq[lam2]*dtau[lam2][tau]
                    )

# Check(ed) that residual imaginary parts are tiny
dsigmaZH = np.real(dsigmaZH)

#pp -> WH

# kinematics
s_hat_min   = (m['H'] + m['W'])**2
s_hat_max   = E_LHC**2
s_hat       = s_hat_min+(s_hat_max-s_hat_min)*np.random.uniform(0,0.005,N_events)
sqrt_s_hat  = np.sqrt(s_hat)
x_min       = np.sqrt( s_hat/s_hat_max )
abs_y_max   = - np.log(x_min)
y           = np.random.uniform(-1,1, N_events)*abs_y_max

w_W         = (s_hat + m['W']**2-m['H']**2)/(2*np.sqrt(s_hat))
k_W         = np.sqrt( w_W**2-m['W']**2 )

x1          = sqrt_s_hat/E_LHC*np.exp(y)
x2          = sqrt_s_hat/E_LHC*np.exp(-y)

# parameters
h1W      = 0.
h2W      = 0.
h3W      = 0.
h4Wtilde = 0.

constWH = 1 
#constWH = 8192*pi**3*Gamma['W']*E_LHC**2

Vud = 1.

dsigmaWH  = np.zeros(N_events).astype('complex128')

ux1     = np.array( [ pdf.pdf( x,  1 ) for x in x1 ] ) 
ubarx1  = np.array( [ pdf.pdf( x, -1 ) for x in x1 ] ) 
ux2     = np.array( [ pdf.pdf( x,  1 ) for x in x2 ] ) 
ubarx2  = np.array( [ pdf.pdf( x, -1 ) for x in x2 ] ) 

dx1     = np.array( [ pdf.pdf( x,  2 ) for x in x1 ] ) 
dbarx1  = np.array( [ pdf.pdf( x, -2 ) for x in x1 ] ) 
dx2     = np.array( [ pdf.pdf( x,  2 ) for x in x2 ] ) 
dbarx2  = np.array( [ pdf.pdf( x, -2 ) for x in x2 ] ) 


dtau = {}
M_lambda_udbar = {}
M_lambda_dbaru = {}
M_lambda_dubar = {}
M_lambda_ubard = {}
for lambda_boson in [+1, -1, 0]:
    if abs(lambda_boson)==1:
        Nhat_prefac = 1./sqrt(2)*g**2 * m['W'] * sqrt_s_hat / (s_hat - m['W']**2) 
        Nhat = {tuple() :       Nhat_prefac*(1.+h1W+h2W*s_hat/m['W']**2+1j*lambda_boson*h4Wtilde*k_W*sqrt_s_hat/m['W']**2),
                ('h1W',):       Nhat_prefac,
                ('h2W',):       Nhat_prefac*s_hat/m['W']**2,
                ('h3W',):       np.zeros(N_events),
                ('h4Wtilde',):  Nhat_prefac*1j*lambda_boson*k_W*sqrt_s_hat/m['W']**2,
               }

        M_lambda_udbar[lambda_boson] =  { k: - (1-lambda_boson*cos_theta)/sqrt(2.)*np.conjugate(Vud)*Nhat[k] for k in Nhat.keys()}
        M_lambda_dbaru[lambda_boson] =  { k:   (1+lambda_boson*cos_theta)/sqrt(2.)*np.conjugate(Vud)*Nhat[k] for k in Nhat.keys()}

        M_lambda_dubar[lambda_boson] =  { k: - (1-lambda_boson*cos_theta)/sqrt(2.)*Vud*Nhat[k] for k in Nhat.keys()}
        M_lambda_ubard[lambda_boson] =  { k:   (1+lambda_boson*cos_theta)/sqrt(2.)*Vud*Nhat[k] for k in Nhat.keys()}
    else:
        Nhat_prefac = - 1./sqrt(2)*g**2 * w_W * sqrt_s_hat / (s_hat - m['W']**2)
        Nhat = {tuple() :       Nhat_prefac*(1.+h1W+h2W*s_hat/m['W']**2+h3W*k_W**2*sqrt_s_hat/(m['W']**2*w_W)),
                ('h1W',):       Nhat_prefac,
                ('h2W',):       Nhat_prefac*s_hat/m['W']**2,
                ('h3W',):       Nhat_prefac*k_W**2*sqrt_s_hat/(m['W']**2*w_W),
                ('h4Wtilde',):  np.zeros(N_events),
                }

        M_lambda_udbar[lambda_boson] = { k: sin_theta*np.conjugate(Vud)*Nhat[k] for k in Nhat.keys()} 
        M_lambda_dbaru[lambda_boson] = { k: M_lambda_udbar[lambda_boson][k] for k in Nhat.keys()}

        M_lambda_dubar[lambda_boson] = { k: sin_theta*Vud*Nhat[k] for k in Nhat.keys()}
        M_lambda_ubard[lambda_boson] = { k: M_lambda_udbar[lambda_boson][k] for k in Nhat.keys()}

    dtau[lambda_boson] = {}
    for tau in [+1, -1]:
        if abs(tau)==1:
            dtau[lambda_boson][tau] = tau*(1+lambda_boson*tau*cos_theta_hat)/sqrt(2.)*np.exp(1j*lambda_boson*phi_hat) 
        else:
            dtau[lambda_boson][tau] = sin_theta_hat 


first_derivatives = [('h1W',), ('h2W',), ('h3W',), ('h4Wtilde',)]
second_derivatives= [ ('h1W','h1W'), ('h2W','h2W'), ('h3W', 'h3W'), ('h4Wtilde', 'h4Wtilde'), \
                 ('h1W','h2W'), ('h1W','h3W'), ('h1W', 'h4Wtilde'), ('h2W', 'h3W'), ('h2W', 'h4Wtilde'), ('h3W', 'h4Wtilde') ]
derivatives   = [ tuple() ] + first_derivatives + second_derivatives

dsigmaWplusH  = {der:np.zeros(N_events).astype('complex128') for der in derivatives}
dsigmaWminusH = {der:np.zeros(N_events).astype('complex128') for der in derivatives}

for lam1 in [+1, -1, 0]:
    for lam2 in [+1, -1, 0]:
        # W+

        dsigmaWplusH_prefac = m['W']*k_W/(constWH*s_hat**1.5)*g**2*2
        dsigmaWplusH[tuple()] += dsigmaWplusH_prefac*(
              ux1*   dbarx2*np.conjugate(dtau[lam1][-1])*np.conjugate(M_lambda_udbar[lam1][tuple()])*M_lambda_udbar[lam2][tuple()]*dtau[lam2][-1]
            + dbarx1*ux2*   np.conjugate(dtau[lam1][-1])*np.conjugate(M_lambda_dbaru[lam1][tuple()])*M_lambda_dbaru[lam2][tuple()]*dtau[lam2][-1]
        )
        for der in first_derivatives:
            dsigmaWplusH[der] += dsigmaWplusH_prefac*(
                  ux1*   dbarx2*np.conjugate(dtau[lam1][-1])*(np.conjugate(M_lambda_udbar[lam1][der])*M_lambda_udbar[lam2][tuple()]+np.conjugate(M_lambda_udbar[lam1][tuple()])*M_lambda_udbar[lam2][der])*dtau[lam2][-1]
                + dbarx1*ux2*   np.conjugate(dtau[lam1][-1])*(np.conjugate(M_lambda_dbaru[lam1][der])*M_lambda_dbaru[lam2][tuple()]+np.conjugate(M_lambda_dbaru[lam1][tuple()])*M_lambda_dbaru[lam2][der])*dtau[lam2][-1]
            )
        for der in second_derivatives:
            dsigmaWplusH[der] += dsigmaWplusH_prefac*(
                  ux1*   dbarx2*np.conjugate(dtau[lam1][-1])*(np.conjugate(M_lambda_udbar[lam1][(der[0],)])*M_lambda_udbar[lam2][(der[1],)]+np.conjugate(M_lambda_udbar[lam1][(der[1],)])*M_lambda_udbar[lam2][(der[0],)])*dtau[lam2][-1]
                + dbarx1*ux2*   np.conjugate(dtau[lam1][-1])*(np.conjugate(M_lambda_dbaru[lam1][(der[0],)])*M_lambda_dbaru[lam2][(der[1],)]+np.conjugate(M_lambda_dbaru[lam1][(der[1],)])*M_lambda_dbaru[lam2][(der[0],)])*dtau[lam2][-1]
            )

        dsigmaWminusH_prefac = m['W']*k_W/(constWH*s_hat**1.5)*g**2*2
        dsigmaWminusH[tuple()] += dsigmaWminusH_prefac*(
              dx1*   ubarx2*np.conjugate(dtau[lam1][-1])*np.conjugate(M_lambda_dubar[lam1][tuple()])*M_lambda_dubar[lam2][tuple()]*dtau[lam2][-1]
            + ubarx1*dx2*   np.conjugate(dtau[lam1][-1])*np.conjugate(M_lambda_ubard[lam1][tuple()])*M_lambda_ubard[lam2][tuple()]*dtau[lam2][-1]
        )
        for der in first_derivatives:
            dsigmaWminusH[der] += dsigmaWminusH_prefac*(
                  dx1*   ubarx2*np.conjugate(dtau[lam1][-1])*(np.conjugate(M_lambda_dubar[lam1][der])*M_lambda_dubar[lam2][tuple()]+np.conjugate(M_lambda_dubar[lam1][tuple()])*M_lambda_dubar[lam2][der])*dtau[lam2][-1]
                + ubarx1*dx2*   np.conjugate(dtau[lam1][-1])*(np.conjugate(M_lambda_ubard[lam1][der])*M_lambda_ubard[lam2][tuple()]+np.conjugate(M_lambda_ubard[lam1][tuple()])*M_lambda_ubard[lam2][der])*dtau[lam2][-1]
            )
        for der in second_derivatives:
            dsigmaWminusH[der] += dsigmaWminusH_prefac*(
                  dx1*   ubarx2*np.conjugate(dtau[lam1][-1])*(np.conjugate(M_lambda_dubar[lam1][(der[0],)])*M_lambda_dubar[lam2][(der[1],)]+np.conjugate(M_lambda_dubar[lam1][(der[1],)])*M_lambda_dubar[lam2][(der[0],)])*dtau[lam2][-1]
                + ubarx1*dx2*   np.conjugate(dtau[lam1][-1])*(np.conjugate(M_lambda_ubard[lam1][(der[0],)])*M_lambda_ubard[lam2][(der[1],)]+np.conjugate(M_lambda_ubard[lam1][(der[1],)])*M_lambda_ubard[lam2][(der[0],)])*dtau[lam2][-1]
            )

# Check(ed) that residual imaginary parts are tiny
dsigmaWplusH  = {k:np.real(dsigmaWplusH[k]) for k in derivatives}
dsigmaWminusH = {k:np.real(dsigmaWminusH) for k in derivatives}
