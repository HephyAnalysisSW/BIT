import ROOT
import numpy as np
import random

from math import sin, cos, sqrt, pi
'''Implement the model from https://arxiv.org/pdf/1706.01816.pdf 
'''

import ROOT
import csv
import os
import array
h_pdf = {} 
for c_pdf in [ "u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar", "b", "bbar", "gluon"]:

    with open(os.path.expandvars("$CMSSW_BASE/src/BIT/VH_models/pdf_data/pdf_%s.txt"%c_pdf)) as f:
        reader = csv.reader(f)
        data = list(reader)
        thresholds = []
        values     = []
        for s_thr, s_val in data:
            thresholds.append(float(s_thr))
            values.append(float(s_val))
        h_pdf[c_pdf] = ROOT.TH1D( c_pdf, c_pdf, len(thresholds), array.array('d', thresholds+[1.001] ) )
        for thr, val in zip(thresholds, values):
            h_pdf[c_pdf].SetBinContent( h_pdf[c_pdf].FindBin( thr ), val ) 

pdg = {1:"d", 2:"u", 3:"s", 4:"c", 5:"b", -1:"dbar", -2:"ubar", -3:"sbar", -4:"cbar", -5:"bbar", 21:"gluon"}

def pdf( x, f ):
    histo = h_pdf[f] if type(f)==str else h_pdf[pdg[f]]
    if x<histo.GetXaxis().GetXmin() or x>1:
        raise RuntimeError("Minimum is %5.5f, maximum is 1, you asked for %5.5f" %( histo.GetXaxis().GetXmin(), x))
    return max(0, histo.Interpolate(x))

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


E_LHC       = 13000
s_hat_max   = E_LHC**2
s_hat_clip  = 0.015

# Qq
Qq  = {1:-1./3., 2: 2./3.}
T3q = {1:-.5,    2:.5}

def ac_basis(**kwargs):
    ''' Eq. 2.3 of https://arxiv.org/pdf/1706.01816.pdf
    '''
    ac_basis = ['aZ', 'bZ', 'cZ', 'bZtilde', 'bgamma', 'cgamma', 'bgammatilde', 'aW', 'bW', 'cW', 'bWtilde']
    default_ac_parameters = { }#'Lambda':1000 }
    default_ac_parameters.update( {var:0. for var in ac_basis} )
    d = { key:val for key, val in default_ac_parameters.iteritems() }
    for key, val in kwargs.iteritems():
        if not key in ac_basis:
            raise RuntimeError ("AC parameter not known.")
        else:
            d[key] = float(val)

    return {'h1Z': d['aZ'] + d['bZ'] - (d['bZ'] - d['cZ'])*m['H']**2/m['Z']**2,
            'h2Z': d['bZ'],
            'h3Z': -2*(d['bZ'] - d['cZ']), 
            'h4Ztilde': -2*d['bZtilde'],
            'h1gamma':0.5*(d['bgamma']-d['cgamma'])*(m['Z']**2-m['H']**2)/m['Z']**2,
            'h2gamma':0.5*(d['bgamma']+d['cgamma']),
            'h3gamma':-(d['bgamma']-d['cgamma']),
            'h4gammatilde':-d['bgammatilde'],
            'h1W':d['aW']+d['bW']-(d['bW']-d['cW'])*m['H']**2/m['W']**2,
            'h2W':d['bW'],
            'h3W':-2*(d['bW']-d['cW']),
            'h4Wtilde':-2*d['bWtilde'],}

# EFT settings, parameters, defaults
wilson_coefficients    = ['h1Z', 'h2Z', 'h3Z', 'h4Ztilde', 'h1gamma', 'h2gamma', 'h3gamma', 'h4gammatilde', 'h1W', 'h2W', 'h3W', 'h4Wtilde']
default_eft_parameters = { }#'Lambda':1000 }
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
sm         = make_eft()

# qq -> ZH
feature_names_ZH =  ['sqrt_s_hat', 'y', 'cos_theta', 'phi_hat', 'cos_theta_hat', 
                    'fLL', 'f1TT', 'f2TT' , 'f1LT', 'f2LT', 'f1tildeLT', 'f2tildeLT', 'fTTprime', 'ftildeTTprime']

def get_ZH_events(N_events=10):
    # theta of boson in the qq restframe
    cos_theta = np.random.uniform(-1,1,N_events)

    phi_hat       = pi*np.random.uniform(-1,1,N_events)
    cos_theta_hat = np.random.uniform(-1,1,N_events)

    # kinematics
    s_hat_min   = (m['H'] + m['Z'])**2
    s_hat       = s_hat_min+(s_hat_max-s_hat_min)*np.random.uniform(0, s_hat_clip, N_events)
    sqrt_s_hat  = np.sqrt(s_hat)

    x_min       = sqrt_s_hat/np.sqrt( s_hat_max ) #at least one of the x must be above this value
    abs_y_max   = - np.log(x_min)
    y           = np.random.uniform(-1,1, N_events)*abs_y_max

    # Eq. 3.6 of Spannowsky (https://arxiv.org/pdf/1912.07628.pdf)
    C2_theta     =  cos_theta**2  
    C2_theta_hat =  cos_theta_hat**2  

    S2_theta     =  1. - C2_theta 
    S2_theta_hat =  1. - C2_theta_hat
    S_theta     = np.sqrt(S2_theta) 
    S_theta_hat = np.sqrt(S2_theta_hat) 
    C_phi_hat   = np.cos(phi_hat)
    S_phi_hat   = np.sin(phi_hat)
    C_2phi_hat   = np.cos(2*phi_hat)
    S_2phi_hat   = np.sin(2*phi_hat)

    fLL         = S2_theta*S2_theta_hat
    f1TT        = cos_theta*cos_theta_hat
    f2TT        = (1+C2_theta)*(1+C2_theta_hat)
    f1LT        = C_phi_hat*S_theta*S_theta_hat
    f2LT        = f1LT*cos_theta*cos_theta_hat 
    f1tildeLT   = S_phi_hat*S_theta*S_theta_hat
    f2tildeLT   = f1tildeLT*cos_theta*cos_theta_hat 
    fTTprime    = C_2phi_hat*fLL
    ftildeTTprime=S_2phi_hat*fLL

    return np.transpose(np.array( [sqrt_s_hat, y, cos_theta, phi_hat, cos_theta_hat, fLL, f1TT, f2TT, f1LT, f2LT, f1tildeLT, f2tildeLT, fTTprime, ftildeTTprime]))

def get_ZH_weights( features, eft):

    sqrt_s_hat    = features[:,feature_names_ZH.index('sqrt_s_hat')]
    y             = features[:,feature_names_ZH.index('y')]
    cos_theta     = features[:,feature_names_ZH.index('cos_theta')]
    phi_hat       = features[:,feature_names_ZH.index('phi_hat')]
    cos_theta_hat = features[:,feature_names_ZH.index('cos_theta_hat')]

    s_hat         = sqrt_s_hat**2
    sin_theta     = np.sqrt( 1. - cos_theta**2 ) 
    sin_theta_hat = np.sqrt( 1. - cos_theta_hat**2 ) 

    w_Z           = (s_hat + m['Z']**2-m['H']**2)/(2*np.sqrt(s_hat))
    k_Z           = np.sqrt( w_Z**2-m['Z']**2 )

    x1          = sqrt_s_hat/E_LHC*np.exp(y)
    x2          = sqrt_s_hat/E_LHC*np.exp(-y)

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
    constZH   = 1 
    N_events  = len(features)
    dsigmaZH  = {der:np.zeros(N_events).astype('complex128') for der in derivatives}

    for pdg_quark in [1, 2]:
        qx1     = np.array( [ pdf( x,  pdg_quark ) for x in x1 ] ) 
        qbarx2  = np.array( [ pdf( x, -pdg_quark ) for x in x2 ] ) 

        qbarx1  = np.array( [ pdf( x, -pdg_quark ) for x in x1 ] ) 
        qx2     = np.array( [ pdf( x,  pdg_quark ) for x in x2 ] ) 
        for sigma_quark in [+1, -1]:
            dtau = {}
            M_lambda_sigma_qbarq = {}
            M_lambda_sigma_qqbar = {}
            for lambda_boson in [+1, -1, 0]:
                if abs(lambda_boson)==1:

                    prefac   = gZ*m['Z']*sqrt_s_hat
                    prefac_1 = prefac * gZsigma[sigma_quark][pdg_quark] / (s_hat - m['Z']**2)
                    prefac_2 = prefac * Qq[pdg_quark]*e/s_hat
                    Mhat = {tuple():            prefac_1*(1.+eft['h1Z']+eft['h2Z']*s_hat/m['Z']**2+1j*lambda_boson*eft['h4Ztilde']*k_Z*sqrt_s_hat/m['Z']**2)\
                                               +prefac_2*(eft['h1gamma']+eft['h2gamma']*s_hat/m['Z']**2+1j*lambda_boson*eft['h4gammatilde']*k_Z*sqrt_s_hat/m['Z']**2),
                            ('h1Z',):           prefac_1,
                            ('h2Z',):           prefac_1*s_hat/m['Z']**2,
                            ('h3Z',):           np.zeros(N_events),
                            ('h4Ztilde',):      prefac_1*1j*lambda_boson*k_Z*sqrt_s_hat/m['Z']**2,
                            ('h1gamma',):       prefac_2,
                            ('h2gamma',):       prefac_2*s_hat/m['Z']**2,
                            ('h3gamma',):       np.zeros(N_events),
                            ('h4gammatilde',):  prefac_2*1j*lambda_boson*k_Z*sqrt_s_hat/m['Z']**2,
                            }
                    M_lambda_sigma_qqbar[lambda_boson] = {k: sigma_quark*(1+sigma_quark*lambda_boson*cos_theta)/sqrt(2.)*Mhat[k] for k in Mhat.keys()} 
                    M_lambda_sigma_qbarq[lambda_boson] = {k:-sigma_quark*(1-sigma_quark*lambda_boson*cos_theta)/sqrt(2.)*Mhat[k] for k in Mhat.keys()}
                else:
                    prefac   = sin_theta*(-gZ)*w_Z*sqrt_s_hat
                    prefac_1 = prefac * gZsigma[sigma_quark][pdg_quark]/(s_hat - m['Z']**2)
                    prefac_2 = prefac * Qq[pdg_quark]*e/s_hat
                    M_lambda_sigma_qqbar[lambda_boson] = \
                            {tuple():prefac_1*(1.+eft['h1Z']+eft['h2Z']*s_hat/m['Z']**2+eft['h3Z']*k_Z**2*sqrt_s_hat/(m['Z']**2*w_Z)) 
                                   + prefac_2*(eft['h1gamma']+eft['h2gamma']*s_hat/m['Z']**2+eft['h3gamma']*k_Z**2*sqrt_s_hat/(m['Z']**2*w_Z)),
                            ('h1Z',):           prefac_1,
                            ('h2Z',):           prefac_1*s_hat/m['Z']**2,
                            ('h3Z',):           prefac_1*k_Z**2*sqrt_s_hat/(m['Z']**2*w_Z),
                            ('h4Ztilde',):      np.zeros(N_events),
                            ('h1gamma',):       prefac_2,
                            ('h2gamma',):       prefac_2*s_hat/m['Z']**2,
                            ('h3gamma',):       k_Z**2*sqrt_s_hat/(m['Z']**2*w_Z),
                            ('h4gammatilde',):  np.zeros(N_events),
                            }
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

                        dsigmaZH_prefac = m['Z']*k_Z/(constZH*s_hat**1.5)*gZtaull[tau]**2*Cf
                        qx1_qbarx2      = qx1*qbarx2
                        qbarx1_qx2      = qbarx1*qx2 
                        dsigmaZH[tuple()] += dsigmaZH_prefac*(
                              qx1_qbarx2*np.conjugate(dtau[lam1][tau])*np.conjugate(M_lambda_sigma_qqbar[lam1][tuple()])*M_lambda_sigma_qqbar[lam2][tuple()]*dtau[lam2][tau]
                            + qbarx1_qx2*np.conjugate(dtau[lam1][tau])*np.conjugate(M_lambda_sigma_qbarq[lam1][tuple()])*M_lambda_sigma_qbarq[lam2][tuple()]*dtau[lam2][tau]
                        )
                        for der in first_derivatives:
                            dsigmaZH[der] += dsigmaZH_prefac*(
                                 qx1_qbarx2*np.conjugate(dtau[lam1][tau])*(
                                    np.conjugate(M_lambda_sigma_qqbar[lam1][der])*M_lambda_sigma_qqbar[lam2][tuple()]*dtau[lam2][tau]
                                   +np.conjugate(M_lambda_sigma_qqbar[lam1][tuple()])*M_lambda_sigma_qqbar[lam2][der]*dtau[lam2][tau])
                               + qbarx1_qx2*np.conjugate(dtau[lam1][tau])*(
                                    np.conjugate(M_lambda_sigma_qbarq[lam1][der])*M_lambda_sigma_qbarq[lam2][tuple()]*dtau[lam2][tau]
                                   +np.conjugate(M_lambda_sigma_qbarq[lam1][tuple()])*M_lambda_sigma_qbarq[lam2][der]*dtau[lam2][tau])
                            )
                        for der in second_derivatives:
                            dsigmaZH[der] += dsigmaZH_prefac*(
                                 qx1_qbarx2*np.conjugate(dtau[lam1][tau])*(
                                    np.conjugate(M_lambda_sigma_qqbar[lam1][(der[0],)])*M_lambda_sigma_qqbar[lam2][(der[1],)]*dtau[lam2][tau]
                                   +np.conjugate(M_lambda_sigma_qqbar[lam1][(der[1],)])*M_lambda_sigma_qqbar[lam2][(der[0],)]*dtau[lam2][tau])
                               + qbarx1_qx2*np.conjugate(dtau[lam1][tau])*(
                                    np.conjugate(M_lambda_sigma_qbarq[lam1][(der[0],)])*M_lambda_sigma_qbarq[lam2][(der[1],)]*dtau[lam2][tau]
                                   +np.conjugate(M_lambda_sigma_qbarq[lam1][(der[1],)])*M_lambda_sigma_qbarq[lam2][(der[0],)]*dtau[lam2][tau])
                            )
    # Check(ed) that residual imaginary parts are tiny
    dsigmaZH  = {k:np.real(dsigmaZH[k])  for k in derivatives}
    return dsigmaZH


feature_names_WH =  ['sqrt_s_hat', 'y', 'cos_theta', 'phi_hat', 'cos_theta_hat', 'lepton_charge']

#pp -> WH
def get_WH_events(N_events):

    # theta of boson in the qq restframe
    cos_theta = np.random.uniform(-1,1,N_events)

    phi_hat = pi*np.random.uniform(-1,1,N_events)
    cos_theta_hat = np.random.uniform(-1,1,N_events)

    # kinematics
    s_hat_min   = (m['H'] + m['W'])**2
    s_hat       = s_hat_min+(s_hat_max-s_hat_min)*np.random.uniform(0, s_hat_clip, N_events)
    sqrt_s_hat  = np.sqrt(s_hat)

    x_min       = np.sqrt( s_hat/s_hat_max )
    abs_y_max   = - np.log(x_min)
    y           = np.random.uniform(-1,1, N_events)*abs_y_max

    lepton_charge = np.random.choice([-1,1], N_events)

    return np.transpose(np.array( [sqrt_s_hat, y, cos_theta, phi_hat, cos_theta_hat, lepton_charge]))

def get_WH_weights(features, eft):

    sqrt_s_hat    = features[:,feature_names_WH.index('sqrt_s_hat')]
    y             = features[:,feature_names_WH.index('y')]
    cos_theta     = features[:,feature_names_WH.index('cos_theta')]
    phi_hat       = features[:,feature_names_WH.index('phi_hat')]
    cos_theta_hat = features[:,feature_names_WH.index('cos_theta_hat')]
    lepton_charge = features[:,feature_names_WH.index('lepton_charge')]

    is_plus       = lepton_charge==1
    s_hat         = sqrt_s_hat**2
    sin_theta     = np.sqrt( 1. - cos_theta**2 ) 
    sin_theta_hat = np.sqrt( 1. - cos_theta_hat**2 ) 

    w_W         = (s_hat + m['W']**2-m['H']**2)/(2*np.sqrt(s_hat))
    k_W         = np.sqrt( w_W**2-m['W']**2 )

    x1          = sqrt_s_hat/E_LHC*np.exp(y)
    x2          = sqrt_s_hat/E_LHC*np.exp(-y)

    constWH = 1 
    #constWH = 8192*pi**3*Gamma['W']*E_LHC**2

    Vud = 1.
    N_events= len(features)
    dsigmaWH= np.zeros(N_events).astype('complex128')

    ux1     = np.array( [ pdf( x,  2 ) for x in x1 ] ) 
    ubarx1  = np.array( [ pdf( x, -2 ) for x in x1 ] ) 
    ux2     = np.array( [ pdf( x,  2 ) for x in x2 ] ) 
    ubarx2  = np.array( [ pdf( x, -2 ) for x in x2 ] ) 

    dx1     = np.array( [ pdf( x,  1 ) for x in x1 ] ) 
    dbarx1  = np.array( [ pdf( x, -1 ) for x in x1 ] ) 
    dx2     = np.array( [ pdf( x,  1 ) for x in x2 ] ) 
    dbarx2  = np.array( [ pdf( x, -1 ) for x in x2 ] ) 

    dtau = {}
    M_lambda_udbar = {}
    M_lambda_dbaru = {}
    M_lambda_dubar = {}
    M_lambda_ubard = {}
    for lambda_boson in [+1, -1, 0]:
        if abs(lambda_boson)==1:
            Nhat_prefac = 1./sqrt(2)*g**2 * m['W'] * sqrt_s_hat / (s_hat - m['W']**2) 
            Nhat = {tuple() :       Nhat_prefac*(1.+eft['h1W']+eft['h2W']*s_hat/m['W']**2+1j*lambda_boson*eft['h4Wtilde']*k_W*sqrt_s_hat/m['W']**2),
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
            Nhat = {tuple() :       Nhat_prefac*(1.+eft['h1W']+eft['h2W']*s_hat/m['W']**2+eft['h3W']*k_W**2*sqrt_s_hat/(m['W']**2*w_W)),
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
    second_derivatives= [('h1W','h1W'), ('h2W','h2W'), ('h3W', 'h3W'), ('h4Wtilde', 'h4Wtilde'),\
                         ('h1W','h2W'), ('h1W','h3W'), ('h1W', 'h4Wtilde'), ('h2W', 'h3W'), ('h2W', 'h4Wtilde'), ('h3W', 'h4Wtilde')]
    derivatives   = [tuple()] + first_derivatives + second_derivatives

    dsigmaWH  = {der:np.zeros(N_events).astype('complex128') for der in derivatives}

    dsigmaWH_prefac = m['W']*k_W/(constWH*s_hat**1.5)*g**2*2 #factor 2 from sum_{u,d} |V_{ud}|**2 = 2
    for lam1 in [+1, -1, 0]:
        for lam2 in [+1, -1, 0]:
            # pp->HW+
            dsigmaWH[tuple()][is_plus] += dsigmaWH_prefac[is_plus]*(
                  ux1[is_plus]*dbarx2[is_plus]*np.conjugate(dtau[lam1][-1][is_plus])*np.conjugate(M_lambda_udbar[lam1][tuple()][is_plus])*M_lambda_udbar[lam2][tuple()][is_plus]*dtau[lam2][-1][is_plus]
                + dbarx1[is_plus]*ux2[is_plus]*np.conjugate(dtau[lam1][-1][is_plus])*np.conjugate(M_lambda_dbaru[lam1][tuple()][is_plus])*M_lambda_dbaru[lam2][tuple()][is_plus]*dtau[lam2][-1][is_plus]
            )
            for der in first_derivatives:
                dsigmaWH[der][is_plus] += dsigmaWH_prefac[is_plus]*(
                      ux1[is_plus]*dbarx2[is_plus]*np.conjugate(dtau[lam1][-1][is_plus])*
                                (np.conjugate(M_lambda_udbar[lam1][der][is_plus])*M_lambda_udbar[lam2][tuple()][is_plus]\
                               + np.conjugate(M_lambda_udbar[lam1][tuple()][is_plus])*M_lambda_udbar[lam2][der][is_plus])*dtau[lam2][-1][is_plus]
                    + dbarx1[is_plus]*ux2[is_plus]*np.conjugate(dtau[lam1][-1][is_plus])*
                                (np.conjugate(M_lambda_dbaru[lam1][der][is_plus])*M_lambda_dbaru[lam2][tuple()][is_plus]\
                               + np.conjugate(M_lambda_dbaru[lam1][tuple()][is_plus])*M_lambda_dbaru[lam2][der][is_plus])*dtau[lam2][-1][is_plus]
                )
            for der in second_derivatives:
                dsigmaWH[der][is_plus] += dsigmaWH_prefac[is_plus]*(
                      ux1[is_plus]*dbarx2[is_plus]*np.conjugate(dtau[lam1][-1][is_plus])*
                                (np.conjugate(M_lambda_udbar[lam1][(der[0],)][is_plus])*M_lambda_udbar[lam2][(der[1],)][is_plus]
                               + np.conjugate(M_lambda_udbar[lam1][(der[1],)][is_plus])*M_lambda_udbar[lam2][(der[0],)][is_plus])*dtau[lam2][-1][is_plus]
                    + dbarx1[is_plus]*ux2[is_plus]*np.conjugate(dtau[lam1][-1][is_plus])*
                                (np.conjugate(M_lambda_dbaru[lam1][(der[0],)][is_plus])*M_lambda_dbaru[lam2][(der[1],)][is_plus]
                               + np.conjugate(M_lambda_dbaru[lam1][(der[1],)][is_plus])*M_lambda_dbaru[lam2][(der[0],)][is_plus])*dtau[lam2][-1][is_plus]
                )
            # pp->HW-
            dsigmaWH[tuple()][~is_plus] += dsigmaWH_prefac[~is_plus]*(
                  dx1[~is_plus]*ubarx2[~is_plus]*np.conjugate(dtau[lam1][-1][~is_plus])*np.conjugate(M_lambda_dubar[lam1][tuple()][~is_plus])*M_lambda_dubar[lam2][tuple()][~is_plus]*dtau[lam2][-1][~is_plus]
                + ubarx1[~is_plus]*dx2[~is_plus]*np.conjugate(dtau[lam1][-1][~is_plus])*np.conjugate(M_lambda_ubard[lam1][tuple()][~is_plus])*M_lambda_ubard[lam2][tuple()][~is_plus]*dtau[lam2][-1][~is_plus]
            )
            for der in first_derivatives:
                dsigmaWH[der][~is_plus] += dsigmaWH_prefac[~is_plus]*(
                      dx1[~is_plus]*ubarx2[~is_plus]*np.conjugate(dtau[lam1][-1][~is_plus])*(np.conjugate(M_lambda_dubar[lam1][der][~is_plus])*M_lambda_dubar[lam2][tuple()][~is_plus]
                                                   + np.conjugate(M_lambda_dubar[lam1][tuple()][~is_plus])*M_lambda_dubar[lam2][der][~is_plus])*dtau[lam2][-1][~is_plus]
                    + ubarx1[~is_plus]*dx2[~is_plus]*np.conjugate(dtau[lam1][-1][~is_plus])*(np.conjugate(M_lambda_ubard[lam1][der][~is_plus])*M_lambda_ubard[lam2][tuple()][~is_plus]
                                                   + np.conjugate(M_lambda_ubard[lam1][tuple()][~is_plus])*M_lambda_ubard[lam2][der][~is_plus])*dtau[lam2][-1][~is_plus]
                )
            for der in second_derivatives:
                dsigmaWH[der][~is_plus] += dsigmaWH_prefac[~is_plus]*(
                      dx1[~is_plus]*ubarx2[~is_plus]*np.conjugate(dtau[lam1][-1][~is_plus])*(np.conjugate(M_lambda_dubar[lam1][(der[0],)][~is_plus])*M_lambda_dubar[lam2][(der[1],)][~is_plus]
                                                   + np.conjugate(M_lambda_dubar[lam1][(der[1],)][~is_plus])*M_lambda_dubar[lam2][(der[0],)][~is_plus])*dtau[lam2][-1][~is_plus]
                    + ubarx1[~is_plus]*dx2[~is_plus]*np.conjugate(dtau[lam1][-1][~is_plus])*(np.conjugate(M_lambda_ubard[lam1][(der[0],)][~is_plus])*M_lambda_ubard[lam2][(der[1],)][~is_plus]
                                                   + np.conjugate(M_lambda_ubard[lam1][(der[1],)][~is_plus])*M_lambda_ubard[lam2][(der[0],)][~is_plus])*dtau[lam2][-1][~is_plus]
                )

    # Check(ed) that residual imaginary parts are tiny
    dsigmaWH  = {k:np.real(dsigmaWH[k])  for k in derivatives}
    return dsigmaWH

    sqrt_s_hat    = features[:,feature_names_WH.index('sqrt_s_hat')]
    y             = features[:,feature_names_WH.index('y')]
    cos_theta     = features[:,feature_names_WH.index('cos_theta')]
    phi_hat       = features[:,feature_names_WH.index('phi_hat')]
    cos_theta_hat = features[:,feature_names_WH.index('cos_theta_hat')]
    lepton_charge = features[:,feature_names_WH.index('lepton_charge')]

Nbins = 50
plot_options = {
    'sqrt_s_hat': {'binning':[Nbins,200,3000],      'tex':"#sqrt{#hat{s}}",},
    'y':          {'binning':[Nbins,-4,4],          'tex':"y",},
    'cos_theta':  {'binning':[Nbins,-1,1],          'tex':"cos(#theta)",},
    'cos_theta_hat': {'binning':[Nbins,-1,1],       'tex':"cos(#hat{#theta})",},
    'phi_hat':    {'binning':[Nbins,-pi,pi],        'tex':"#hat{#phi}",},
    'lep_charge': {'binning':[3,-1,2],              'tex':"charge(l_{W})",},
    'fLL'         : {'binning':[Nbins,0,1],        'tex':'f_{LL}'          ,},
    'f1TT'        : {'binning':[Nbins,-1,1],        'tex':'f_{1TT}'         ,},
    'f2TT'        : {'binning':[Nbins, 0,4],        'tex':'f_{2TT}'         ,},
    'f1LT'        : {'binning':[Nbins,-1,1],        'tex':'f_{1LT}'         ,},
    'f2LT'        : {'binning':[Nbins,-1,1],        'tex':'f_{2LT}'         ,},
    'f1tildeLT'   : {'binning':[Nbins,-1,1],        'tex':'#tilde{f}_{1LT}' ,},
    'f2tildeLT'   : {'binning':[Nbins,-1,1],        'tex':'#tilde{f}_{2LT}' ,},
    'fTTprime'    : {'binning':[Nbins,-1,1],        'tex':'f_{TT}'     ,},
    'ftildeTTprime':{'binning':[Nbins,-1,1],        'tex':'#tilde{f}_{TT}',},
    } 
