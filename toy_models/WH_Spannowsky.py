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

    with open(os.path.expandvars("$CMSSW_BASE/src/BIT/toy_models/pdf_data/pdf_%s.txt"%c_pdf)) as f:
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
v   = 246.22
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

# EFT settings, parameters, defaults
wilson_coefficients    = ['cHW', 'cHWtil', 'cHQ3']
default_eft_parameters = { 'Lambda':1000. }
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

feature_names =  ['sqrt_s_hat', 'y', 'cos_theta', 'phi_hat', 'cos_theta_hat', 'lepton_charge',
                  'fLL', 'f1TT', 'f2TT' , 'f1LT', 'f2LT', 'f1tildeLT', 'f2tildeLT', 'fTTprime', 'ftildeTTprime']

#pp -> WH
def get_events(N_events):

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

    return np.transpose(np.array( [sqrt_s_hat, y, cos_theta, phi_hat, cos_theta_hat, lepton_charge, fLL, f1TT, f2TT, f1LT, f2LT, f1tildeLT, f2tildeLT, fTTprime, ftildeTTprime]))

def get_weights(features, eft):

    sqrt_s_hat    = features[:,feature_names.index('sqrt_s_hat')]
    y             = features[:,feature_names.index('y')]
    cos_theta     = features[:,feature_names.index('cos_theta')]
    phi_hat       = features[:,feature_names.index('phi_hat')]
    cos_theta_hat = features[:,feature_names.index('cos_theta_hat')]
    lepton_charge = features[:,feature_names.index('lepton_charge')]

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

    N_events= len(features)
    dsigmaWH= np.zeros(N_events).astype('complex128')

    ux1     = np.array( [ pdf( x,  1 ) for x in x1 ] ) 
    ubarx1  = np.array( [ pdf( x, -1 ) for x in x1 ] ) 
    ux2     = np.array( [ pdf( x,  1 ) for x in x2 ] ) 
    ubarx2  = np.array( [ pdf( x, -1 ) for x in x2 ] ) 

    dx1     = np.array( [ pdf( x,  2 ) for x in x1 ] ) 
    dbarx1  = np.array( [ pdf( x, -2 ) for x in x1 ] ) 
    dx2     = np.array( [ pdf( x,  2 ) for x in x2 ] ) 
    dbarx2  = np.array( [ pdf( x, -2 ) for x in x2 ] ) 

    dtau = {}
    M_lambda_udbar = {}
    M_lambda_dbaru = {}
    M_lambda_dubar = {}
    M_lambda_ubard = {}
    for lambda_boson in [+1, -1, 0]:
        if abs(lambda_boson)==1:
            prefac = g**2/sqrt_2*m['W']/sqrt_s_hat 
            M    = {tuple()   :     prefac*(1.+s_hat/m['W']**2*v**2/eft['Lambda']**2*(eft['cHQ3'] + eft['cHW'] - 1j*eft['cHWtil'])),
                   ('cHQ3',)  :     prefac*s_hat/m['W']**2*v**2/eft['Lambda']**2,
                   ('cHW',)   :     prefac*s_hat/m['W']**2*v**2/eft['Lambda']**2,
                   ('cHWtil',): -1j*prefac*s_hat/m['W']**2*v**2/eft['Lambda']**2,
                   }

            M_lambda_udbar[lambda_boson] =  { k: - (1-lambda_boson*cos_theta)/sqrt(2.)*M[k] for k in M.keys()}
            M_lambda_dbaru[lambda_boson] =  { k:   (1+lambda_boson*cos_theta)/sqrt(2.)*M[k] for k in M.keys()}

            M_lambda_dubar[lambda_boson] =  { k: - (1-lambda_boson*cos_theta)/sqrt(2.)*M[k] for k in M.keys()}
            M_lambda_ubard[lambda_boson] =  { k:   (1+lambda_boson*cos_theta)/sqrt(2.)*M[k] for k in M.keys()}
        else:
            prefac = g**2/sqrt_2
            M = {tuple() :     prefac*(1.+2*v**2/eft['Lambda']**2*(2*eft['cHW']+eft['cHQ3']*(-.5+s_hat/(2*m['W']**2)))),
                ('cHQ3',):     prefac*2*v**2/eft['Lambda']**2*(-.5+s_hat/(2*m['W']**2)),
                ('cHW',) :     prefac*2*v**2/eft['Lambda']**2*2,
                ('cHWtil',):   np.zeros(N_events),
                    }

            M_lambda_udbar[lambda_boson] = { k: -sin_theta/2.*M[k] for k in M.keys()} 
            M_lambda_dbaru[lambda_boson] = { k: M_lambda_udbar[lambda_boson][k] for k in M.keys()}

            M_lambda_dubar[lambda_boson] = { k: -sin_theta/2*M[k] for k in M.keys()}
            M_lambda_ubard[lambda_boson] = { k: M_lambda_udbar[lambda_boson][k] for k in M.keys()}

        dtau[lambda_boson] = {}
        for tau in [+1, -1]:
            if abs(tau)==1:
                dtau[lambda_boson][tau] = tau*(1+lambda_boson*tau*cos_theta_hat)/sqrt(2.)*np.exp(1j*lambda_boson*phi_hat) 
            else:
                dtau[lambda_boson][tau] = sin_theta_hat 


    first_derivatives = [('cHQ3',), ('cHW',), ('cHWtil',)]
    second_derivatives= [('cHQ3','cHQ3'), ('cHW','cHW'), ('cHWtil', 'cHWtil'), ('cHQ3','cHW'), ('cHQ3','cHWtil'), ('cHW', 'cHWtil'), ]
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
