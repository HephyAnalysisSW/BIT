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
s_hat_clip  = 0.03

# Qq
Qq  = {1:-1./3., 2: 2./3.}
T3q = {1:-.5,    2:.5}

# EFT settings, parameters, defaults
wilson_coefficients    = ['cHW', 'cHWtil', 'cHQ3']
default_eft_parameters = { 'Lambda':1000. }
default_eft_parameters.update( {var:0. for var in wilson_coefficients} )

first_derivatives = [('cHQ3',), ('cHW',), ('cHWtil',)]
second_derivatives= [('cHQ3','cHQ3'), ('cHW','cHW'), ('cHWtil', 'cHWtil'), ('cHQ3','cHW'), ('cHQ3','cHWtil'), ('cHW', 'cHWtil'), ]
derivatives       = [tuple()] + first_derivatives + second_derivatives

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

feature_names =  ['sqrt_s_hat', 'pT', 'y', 'cos_theta', 'phi_hat', 'cos_theta_hat', 'lepton_charge',
                  'fLL', 'f1TT', 'f2TT' , 'f1LT', 'f2LT', 'f1tildeLT', 'f2tildeLT', 'fTTprime', 'ftildeTTprime']

def make_s_cos_theta( N_events, pT_min = 300):

    # theta of boson in the qq restframe
    cos_theta = np.random.uniform(-1,1,N_events)

    # kinematics
    #s_hat_min   = (m['H'] + m['W'])**2
    s_hat_min   = (2*pT_min)**2 #minimum s_hat for sin(theta)=1
    s_hat       = s_hat_min+(s_hat_max-s_hat_min)*np.random.uniform(0, s_hat_clip, N_events)
    sqrt_s_hat  = np.sqrt(s_hat)

    pT          = 0.5*sqrt_s_hat*np.sqrt(1-cos_theta**2)      
    
    return s_hat, sqrt_s_hat, pT, cos_theta 

#pp -> WH
def getEvents(N_events_requested):

    pT_min = 300

    # correct efficiency of pt cut
    _, _, pT, _ = make_s_cos_theta( N_events_requested, pT_min = pT_min)
    N_events_corrected = int(round(N_events_requested/(np.count_nonzero(pT>pT_min)/float(N_events_requested))))

    # simulate events and compensate pT efficiency
    s_hat, sqrt_s_hat, pT, cos_theta = make_s_cos_theta( N_events_corrected, pT_min = pT_min)

    # apply selection 
    sel = pT>pT_min
    s_hat, sqrt_s_hat, pT, cos_theta = s_hat[sel], sqrt_s_hat[sel], pT[sel], cos_theta[sel]
    N_events = len(s_hat) 
    # remind myself
    print("Requested %i events. Simulated %i events and %i survive pT_min cut of %i." %( N_events_requested, N_events_corrected, N_events, pT_min ) ) 

    phi_hat = pi*np.random.uniform(-1,1,N_events)
    cos_theta_hat = np.random.uniform(-1,1,N_events)

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

    return np.transpose(np.array( [sqrt_s_hat, pT, y, cos_theta, phi_hat, cos_theta_hat, lepton_charge, fLL, f1TT, f2TT, f1LT, f2LT, f1tildeLT, f2tildeLT, fTTprime, ftildeTTprime]))

def getWeights(features, eft):

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

    ux1     = np.array( [ pdf( x,  2 ) for x in x1 ] ) 
    ubarx1  = np.array( [ pdf( x, -2 ) for x in x1 ] ) 
    ux2     = np.array( [ pdf( x,  2 ) for x in x2 ] ) 
    ubarx2  = np.array( [ pdf( x, -2 ) for x in x2 ] ) 

    dx1     = np.array( [ pdf( x,  1 ) for x in x1 ] ) 
    dbarx1  = np.array( [ pdf( x, -1 ) for x in x1 ] ) 
    dx2     = np.array( [ pdf( x,  1 ) for x in x2 ] ) 
    dbarx2  = np.array( [ pdf( x, -1 ) for x in x2 ] ) 

    dtau = {}
    dtau_flipped = {}
    M_lambda_udbar = {}
    M_lambda_dbaru = {}
    M_lambda_dubar = {}
    M_lambda_ubard = {}
    patch = 1#./sqrt_2 #FIXME 
    for lambda_boson in [+1, -1, 0]:
        if abs(lambda_boson)==1:
            prefac = g**2/sqrt_2*m['W']/sqrt_s_hat 
            M    = {tuple()   :   prefac*(1. + s_hat/m['W']**2*v**2/eft['Lambda']**2*(eft['cHQ3'] + eft['cHW'] - 1j*lambda_boson*eft['cHWtil'])),
                   ('cHQ3',)  :   prefac*s_hat/m['W']**2*v**2/eft['Lambda']**2,
                   ('cHW',)   :   prefac*s_hat/m['W']**2*v**2/eft['Lambda']**2,
                   ('cHWtil',): -1j*lambda_boson*prefac*s_hat/m['W']**2*v**2/eft['Lambda']**2,
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

            M_lambda_dubar[lambda_boson] = { k: -sin_theta/2.*M[k] for k in M.keys()}
            M_lambda_ubard[lambda_boson] = { k: M_lambda_udbar[lambda_boson][k] for k in M.keys()}

        dtau[lambda_boson]         = {}
        dtau_flipped[lambda_boson] = {}
        for tau in [+1, -1]:
            if abs(tau)==1:
                dtau[lambda_boson][tau]         = tau*(1+lambda_boson*tau*cos_theta_hat)/sqrt(2.)*np.exp(1j*lambda_boson*phi_hat) 
                dtau_flipped[lambda_boson][tau] = tau*(1+lambda_boson*tau*cos_theta_hat)/sqrt(2.)*np.exp(1j*lambda_boson*(pi-phi_hat)) 
            else:
                dtau[lambda_boson][tau] = sin_theta_hat 
                dtau_flipped[lambda_boson][tau] = dtau[lambda_boson][tau]

    dsigmaWH = {der:np.zeros(N_events).astype('complex128') for der in derivatives}

    dsigmaWH_prefac = m['W']*k_W/(constWH*s_hat**1.5)*g**2*2 #factor 2 from sum_{u,d} |V_{ud}|**2 = 2
    for dtau_ in [ dtau, dtau_flipped]: # Summing up ambiguity from W reco. See e.g. Wulzer https://arxiv.org/pdf/2007.10356.pdf Eq. 21
        for lam1 in [+1, -1, 0]:
            for lam2 in [+1, -1, 0]:
                # pp->HW+
                dsigmaWH[tuple()][is_plus] += dsigmaWH_prefac[is_plus]*(
                      ux1[is_plus]*dbarx2[is_plus]*np.conjugate(dtau_[lam1][-1][is_plus])*np.conjugate(M_lambda_udbar[lam1][tuple()][is_plus])*M_lambda_udbar[lam2][tuple()][is_plus]*dtau_[lam2][-1][is_plus]
                    + dbarx1[is_plus]*ux2[is_plus]*np.conjugate(dtau_[lam1][-1][is_plus])*np.conjugate(M_lambda_dbaru[lam1][tuple()][is_plus])*M_lambda_dbaru[lam2][tuple()][is_plus]*dtau_[lam2][-1][is_plus]
                )
                for der in first_derivatives:
                    dsigmaWH[der][is_plus] += dsigmaWH_prefac[is_plus]*(
                          ux1[is_plus]*dbarx2[is_plus]*np.conjugate(dtau_[lam1][-1][is_plus])*
                                    (np.conjugate(M_lambda_udbar[lam1][der][is_plus])*M_lambda_udbar[lam2][tuple()][is_plus]\
                                   + np.conjugate(M_lambda_udbar[lam1][tuple()][is_plus])*M_lambda_udbar[lam2][der][is_plus])*dtau_[lam2][-1][is_plus]
                        + dbarx1[is_plus]*ux2[is_plus]*np.conjugate(dtau_[lam1][-1][is_plus])*
                                    (np.conjugate(M_lambda_dbaru[lam1][der][is_plus])*M_lambda_dbaru[lam2][tuple()][is_plus]\
                                   + np.conjugate(M_lambda_dbaru[lam1][tuple()][is_plus])*M_lambda_dbaru[lam2][der][is_plus])*dtau_[lam2][-1][is_plus]
                    )
                for der in second_derivatives:
                    dsigmaWH[der][is_plus] += dsigmaWH_prefac[is_plus]*(
                          ux1[is_plus]*dbarx2[is_plus]*np.conjugate(dtau_[lam1][-1][is_plus])*
                                    (np.conjugate(M_lambda_udbar[lam1][(der[0],)][is_plus])*M_lambda_udbar[lam2][(der[1],)][is_plus]
                                   + np.conjugate(M_lambda_udbar[lam1][(der[1],)][is_plus])*M_lambda_udbar[lam2][(der[0],)][is_plus])*dtau_[lam2][-1][is_plus]
                        + dbarx1[is_plus]*ux2[is_plus]*np.conjugate(dtau_[lam1][-1][is_plus])*
                                    (np.conjugate(M_lambda_dbaru[lam1][(der[0],)][is_plus])*M_lambda_dbaru[lam2][(der[1],)][is_plus]
                                   + np.conjugate(M_lambda_dbaru[lam1][(der[1],)][is_plus])*M_lambda_dbaru[lam2][(der[0],)][is_plus])*dtau_[lam2][-1][is_plus]
                    )
                # pp->HW-
                dsigmaWH[tuple()][~is_plus] += dsigmaWH_prefac[~is_plus]*(
                      dx1[~is_plus]*ubarx2[~is_plus]*np.conjugate(dtau_[lam1][-1][~is_plus])*np.conjugate(M_lambda_dubar[lam1][tuple()][~is_plus])*M_lambda_dubar[lam2][tuple()][~is_plus]*dtau_[lam2][-1][~is_plus]
                    + ubarx1[~is_plus]*dx2[~is_plus]*np.conjugate(dtau_[lam1][-1][~is_plus])*np.conjugate(M_lambda_ubard[lam1][tuple()][~is_plus])*M_lambda_ubard[lam2][tuple()][~is_plus]*dtau_[lam2][-1][~is_plus]
                )
                for der in first_derivatives:
                    dsigmaWH[der][~is_plus] += dsigmaWH_prefac[~is_plus]*(
                          dx1[~is_plus]*ubarx2[~is_plus]*np.conjugate(dtau_[lam1][-1][~is_plus])*(np.conjugate(M_lambda_dubar[lam1][der][~is_plus])*M_lambda_dubar[lam2][tuple()][~is_plus]
                                                       + np.conjugate(M_lambda_dubar[lam1][tuple()][~is_plus])*M_lambda_dubar[lam2][der][~is_plus])*dtau_[lam2][-1][~is_plus]
                        + ubarx1[~is_plus]*dx2[~is_plus]*np.conjugate(dtau_[lam1][-1][~is_plus])*(np.conjugate(M_lambda_ubard[lam1][der][~is_plus])*M_lambda_ubard[lam2][tuple()][~is_plus]
                                                       + np.conjugate(M_lambda_ubard[lam1][tuple()][~is_plus])*M_lambda_ubard[lam2][der][~is_plus])*dtau_[lam2][-1][~is_plus]
                    )
                for der in second_derivatives:
                    dsigmaWH[der][~is_plus] += dsigmaWH_prefac[~is_plus]*(
                          dx1[~is_plus]*ubarx2[~is_plus]*np.conjugate(dtau_[lam1][-1][~is_plus])*(np.conjugate(M_lambda_dubar[lam1][(der[0],)][~is_plus])*M_lambda_dubar[lam2][(der[1],)][~is_plus]
                                                       + np.conjugate(M_lambda_dubar[lam1][(der[1],)][~is_plus])*M_lambda_dubar[lam2][(der[0],)][~is_plus])*dtau_[lam2][-1][~is_plus]
                        + ubarx1[~is_plus]*dx2[~is_plus]*np.conjugate(dtau_[lam1][-1][~is_plus])*(np.conjugate(M_lambda_ubard[lam1][(der[0],)][~is_plus])*M_lambda_ubard[lam2][(der[1],)][~is_plus]
                                                       + np.conjugate(M_lambda_ubard[lam1][(der[1],)][~is_plus])*M_lambda_ubard[lam2][(der[0],)][~is_plus])*dtau_[lam2][-1][~is_plus]
                    )

    # Check(ed) that residual imaginary parts are tiny
    dsigmaWH  = {k:np.real(dsigmaWH[k])  for k in derivatives}
    return dsigmaWH

Nbins = 50
plot_options = {
    'sqrt_s_hat': {'binning':[40,650,2250],      'tex':"#sqrt{#hat{s}}",},
    'pT':         {'binning':[35,300,1000],      'tex':"p_{T}",},
    'y':          {'binning':[Nbins,-4,4],          'tex':"y",},
    'cos_theta':  {'binning':[Nbins,-1,1],          'tex':"cos(#theta)",},
    'cos_theta_hat': {'binning':[Nbins,-1,1],       'tex':"cos(#hat{#theta})",},
    'phi_hat':    {'binning':[Nbins,-pi,pi],        'tex':"#hat{#phi}",},
    'lepton_charge': {'binning':[3,-1,2],              'tex':"charge(l_{W})",},
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

eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},
    {'color':ROOT.kBlue+2,      'eft':make_eft(cHQ3=.05),  'tex':"c_{HQ}^{(3)}=0.05"},
    {'color':ROOT.kBlue-4,      'eft':make_eft(cHQ3=-.05), 'tex':"c_{HQ}^{(3)}=-0.05"},
    {'color':ROOT.kGreen+2,     'eft':make_eft(cHW=0.5),   'tex':"c_{HW}=0.5"},
    {'color':ROOT.kGreen-4,     'eft':make_eft(cHW=-0.5),  'tex':"c_{HW}=-0.5"},
    {'color':ROOT.kMagenta+2,   'eft':make_eft(cHWtil=0.5),   'tex':"c_{H#tilde{W}}=0.5"},
    {'color':ROOT.kMagenta-4,   'eft':make_eft(cHWtil=-0.5),  'tex':"c_{H#tilde{W}}=-0.5"},
]

bit_cfg = {der: {'n_trees': 250,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 30,} for der in derivatives if der!=tuple() }
bit_cfg[('cHQ3',)]['n_trees'] = 80
bit_cfg[('cHQ3','cHQ3')]['n_trees'] = 80

def load(directory = '/groups/hephy/cms/robert.schoefbeck/BIT/models/', prefix = 'bit_WH_Spannowsky_nTraining_2000000', derivatives=derivatives):
    import sys, os
    sys.path.insert(0,os.path.expandvars("$CMSSW_BASE/src/BIT"))
    from BoostedInformationTree import BoostedInformationTree
    bits = {}
    for derivative in derivatives:
        if derivative == tuple(): continue

        filename = os.path.expandvars(os.path.join(directory, "%s_derivative_%s"% (prefix, '_'.join(derivative))) + '.pkl')
        try:
            print ("Loading %s for %r"%( filename, derivative))
            bits[derivative] = BoostedInformationTree.load(filename)
        except IOError:
            print ("Could not load %s for %r"%( filename, derivative))

    return bits 
