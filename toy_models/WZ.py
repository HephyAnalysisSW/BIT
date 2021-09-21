import ROOT
import numpy as np
import random
import array
from math import sin, cos, sqrt, pi, log
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
gL = -g*(1-2*s2w)/(2.*cw)
gR = g*s2w/cw

# EFT settings, parameters, defaults
wilson_coefficients    = ['c3PQ', 'cW']
observables = ["s", "Theta", 'thetaW', 'phiW', 'thetaZ', 'phiZ', 'pTZW' ]
default_eft_parameters = { 'Lambda':1000 }
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

# Eq. 17 and derivatives
def M(h1, h2, s, Theta, lep_w_charge, eft, der=None):

        if h1==h2==0:
            if der is None:
                return (-g**2/(2*sqrt(2))-sqrt(2)*eft['c3PQ']*s/eft['Lambda']**2)*np.sin(lep_w_charge*Theta)
            elif der=='c3PQ':
                return                                -sqrt(2)*s/eft['Lambda']**2*np.sin(lep_w_charge*Theta)
            else:
                return 0.
        elif h1==h2==1 or h1==h2==-1:
            if der is None:
                return 3*g*cw*eft['cW']*s/eft['Lambda']**2*np.sin(lep_w_charge*Theta)/sqrt(2)
            elif der=='cW':
                return 3*g*cw*s/eft['Lambda']**2*np.sin(lep_w_charge*Theta)/sqrt(2)
            else:
                return 0.
        elif h1==-1 and h2==1:
            if der is not None: return 0.
            return -g**2*(lep_w_charge*s2w-3*c2w*np.cos(lep_w_charge*Theta))/(3*sqrt(2)*cw)/np.tan(0.5*lep_w_charge*Theta)
        elif h1==1 and h2==-1:
            if der is not None: return 0.
            return  g**2*(lep_w_charge*s2w-3*c2w*np.cos(lep_w_charge*Theta))/(3*sqrt(2)*cw)*np.tan(0.5*lep_w_charge*Theta)
        else:
            return 0.


# pdf(x1) * pdf( x2 ) histo for x1 x2 S > (2*pt_threshold)^2, pt_threshold = 300

import csv

pt_threshold = 300.
E_LHC        = 13000.
b = 0.97
a = (2*pt_threshold)**2/(E_LHC**2*b)
Nspace = 1000
delta = (log(b) - log(a))/Nspace
logRange = np.exp( np.arange( log(a), log(b) + 2*delta, delta))

h_pdf_dbar_u = ROOT.TH2D("h_pdf_dbar_u", "h_pdf_dbar_u", len(logRange)-1, array.array('d', logRange), len(logRange)-1, array.array('d', logRange))
with open('/eos/vbc/user/robert.schoefbeck/TMB/pdf_data/WZ_x1x2_dbar_u.txt') as f:
    reader = csv.reader(f)
    data = list(reader)

    for x1, x2, value in data:
        x1, x2, value = float(x1), float(x2), float(value)
        i_bin = h_pdf_dbar_u.FindBin(x1,x2)
        #print x1, x2, i_bin, value
        h_pdf_dbar_u.SetBinContent( i_bin, value)
    del data

h_pdf_ubar_d = ROOT.TH2D("h_pdf_ubar_d", "h_pdf_ubar_d", len(logRange)-1, array.array('d', logRange), len(logRange)-1, array.array('d', logRange))
vals = []
with open('/eos/vbc/user/robert.schoefbeck/TMB/pdf_data/WZ_x1x2_ubar_d.txt') as f:
    reader = csv.reader(f)
    data = list(reader)

    for x1, x2, value in data:
        x1_, x2_, value_ = float(x1), float(x2), float(value)
        i_bin = h_pdf_ubar_d.FindBin(x1_,x2_)
        h_pdf_ubar_d.SetBinContent( i_bin, value_)
        #h_pdf_ubar_d.Fill( x1_, x2_, value_)
        vals.append([x1_,x2_,value_])
    del data

#import Analysis.Tools.syncer as syncer
#c1 = ROOT.TCanvas()
#h_pdf_dbar_u.Draw('COLZ')
#c1.SetLogz()
#ROOT.c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/h_pdf_dbar_u.png")
#h_pdf_ubar_d.Draw('COLZ')
#c1.SetLogz()
#ROOT.c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/h_pdf_ubar_d.png")

def get_s_Theta(nEvents):
    # generate theta first for pT(Z)>300 -> this means that a few very forward Z events may end up outside the LHC reach
    #cosTheta_max = cos(np.arcsin((2*pt_threshold/E_LHC)**2))
    #cosTheta_min = -cosTheta_max

    cosTheta_max = 0.95
    cosTheta_min = -0.95
    
    Theta   = np.arccos( cosTheta_min+(cosTheta_max-cosTheta_min)*np.random.random(nEvents) ) # no phase space factors from two-particle phase space when cos(theta) is flat
    
    min_x1  = (2*pt_threshold/np.sin(Theta))**2/(E_LHC**2)
    # x1 values scaled to [a,1] interval
    x1      = min_x1 + (1-min_x1)*np.random.random(nEvents)
    min_x2  = (2*pt_threshold/np.sin(Theta))**2/(E_LHC**2*x1)
    x2      = min_x2 + (1-min_x2)*np.random.random(nEvents)

    # some of the CTEQ PDFs are negative above this value
    x2[x2>0.97] = 0.97
    x1[x1>0.97] = 0.97

    shuffle       = np.random.randint(2,size=nEvents).astype(bool)
    lep_w_charge  = np.random.choice([-1,1],size=nEvents)

    w_pdf = []
    for i_x in range(nEvents):
        #print x1_, x2_, h_pdf.GetBinContent( h_pdf.FindBin(x1_, x2_))
        if shuffle[i_x]:
            x1_, x2_ = x2[i_x], x1[i_x]
        else:
            x1_, x2_ = x1[i_x], x2[i_x]
        if lep_w_charge[i_x]==1: 
            h_pdf = h_pdf_dbar_u
        else:
            h_pdf = h_pdf_ubar_d
        if h_pdf.GetBinContent( h_pdf.FindBin( x1_, x2_ ))<=0:
            print lep_w_charge[i_x], x1_, x2_, h_pdf.GetBinContent( h_pdf.FindBin( x1_, x2_ )),"s", sqrt(x1_*x2_*13000**2)
        w_pdf.append( h_pdf.GetBinContent( h_pdf.FindBin( x1_, x2_ )))

    assert len(x1)==nEvents and len(x2)==nEvents, "Not enough events produced."

    return x1*x2*13000**2, Theta, np.array(w_pdf), lep_w_charge, x1, x2

# arxiv:1708.07823
Wigner_d = {
    -1: lambda angle: 0.5*(1.-np.cos(angle)),
     0: lambda angle: 1./sqrt(2)*np.sin(angle),
     1: lambda angle: 0.5*(1.+np.cos(angle)),
    }

class WZ:
    def __init__( self, eft ):
        self.eft = eft
        self.feature_names = ['Theta', 'phiW', 'phiZ', 'thetaW', 'thetaZ'] 

    def getEvents( self, nEvents ): 
        # production
        s, Theta, w_pdf, lep_w_charge, x1, x2 = get_s_Theta(nEvents)

        kaellen = s**2 + m['W']**2 + m['Z']**2 - 2*s*(m['W'] + m['Z']) - 2*m['W']*m['Z']

        phiW    = 2*pi*np.random.random(nEvents)
        phiZ    = 2*pi*np.random.random(nEvents)
        thetaW  = np.arccos( -1+2*np.random.random(nEvents) )
        thetaZ  = np.arccos( -1+2*np.random.random(nEvents) )

        features = np.transpose(np.array( [Theta, phiW, phiZ,  thetaW, thetaZ, lep_w_charge]))

        weights = { key :np.zeros( len(features) ) for key in [ (), ('cW',), ('c3PQ',), ('cW', 'cW'), ('c3PQ', 'cW'), ('c3PQ','c3PQ')] }

        prefac = m['W']*m['Z']/(24*Gamma['W']*Gamma['Z']*s)
        phase_space = np.sqrt(kaellen)/(8.*s)                    
        for hZ in [-1,0,1]:
            for hZp in [-1,0,1]:
                for hW in [-1,0,1]:
                    for hWp in [-1,0,1]:
                        facs = w_pdf*prefac*phase_space*\
                            gW**2 * np.cos(0.5*(hW-hWp)*(pi-2*phiW))\
                            * Wigner_d[hW](thetaW)*Wigner_d[hWp](thetaW)\
                            *(np.cos(0.5*pi*(hW-hWp)+(hZ-hZp)*(pi+phiZ)*gR**2*Wigner_d[hZ](pi-thetaZ)*Wigner_d[hZp](pi-thetaZ))\
                             +np.cos(0.5*pi*(hW-hWp)+(hZ-hZp)*phiZ)*gL**2*Wigner_d[hZ](thetaZ)*Wigner_d[hZp](thetaZ)) 

                        M1      = M(hZ,  hW,  s, Theta, lep_w_charge, self.eft)
                        M2      = M(hZp, hWp, s, Theta, lep_w_charge, self.eft)
                        M1_c3PQ = M(hZ,  hW,  s, Theta, lep_w_charge, self.eft, der='c3PQ')
                        M2_c3PQ = M(hZp, hWp, s, Theta, lep_w_charge, self.eft, der='c3PQ')
                        M1_cW   = M(hZ,  hW,  s, Theta, lep_w_charge, self.eft, der='cW')
                        M2_cW   = M(hZp, hWp, s, Theta, lep_w_charge, self.eft, der='cW')

                        M1n     = M(hZ,  hW,  s, -Theta, lep_w_charge, self.eft)
                        M2n     = M(hZp, hWp, s, -Theta, lep_w_charge, self.eft)
                        M1n_c3PQ= M(hZ,  hW,  s, -Theta, lep_w_charge, self.eft, der='c3PQ')
                        M2n_c3PQ= M(hZp, hWp, s, -Theta, lep_w_charge, self.eft, der='c3PQ')
                        M1n_cW  = M(hZ,  hW,  s, -Theta, lep_w_charge, self.eft, der='cW')
                        M2n_cW  = M(hZp, hWp, s, -Theta, lep_w_charge, self.eft, der='cW')

                        M_prods             = M1*M2 + M1n*M2n 
                        M_prods_c3PQ        = M1_c3PQ*M2 + M1*M2_c3PQ + M1n_c3PQ*M2n + M1n*M2n_c3PQ
                        M_prods_c3PQ_c3PQ   = 2*M1_c3PQ*M2_c3PQ + 2*M1n_c3PQ*M2n_c3PQ
                        M_prods_cW          = M1_cW*M2 + M1*M2_cW + M1n_cW*M2n + M1n*M2n_cW
                        M_prods_cW_cW       = 2*M1_cW*M2_cW + 2*M1n_cW*M2n_cW
                        M_prods_c3PQ_cW     = M1_c3PQ*M2_cW + M1_cW*M2_c3PQ + M1n_c3PQ*M2n_cW + M1n_cW*M2n_c3PQ
    
                        weights[()]         += M_prods*facs
                        weights[('c3PQ',)]  += M_prods_c3PQ*facs
                        weights[('cW',)]    += M_prods_cW*facs
                        weights[('c3PQ','c3PQ')]  += M_prods_c3PQ_c3PQ*facs
                        weights[('cW','cW')]      += M_prods_cW_cW*facs
                        weights[('c3PQ','cW')]    += M_prods_c3PQ_cW*facs

        return features, weights

if __name__=="__main__":
    wz = WZ(random_eft)
