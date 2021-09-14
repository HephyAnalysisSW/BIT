#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
from math import log, exp
import copy

# RootTools
from RootTools.core.standard   import *

# Analysis
import Analysis.Tools.syncer

# BIT
from BoostedInformationTree import BoostedInformationTree

# User
from user import plot_directory as user_plot_directory

# Model choices
allModels = set( [ os.path.splitext(item)[0] for item in os.listdir( "toy_models" ) if not item.startswith("_") ] )

# Models an theta values
# exponential - 0.001
# power_law   - 0.2

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="BIT_v1",                                                        help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="mixture", type=str,   choices=allModels,                    help="import model")
argParser.add_argument("--nTraining",          action="store",      default=500000,        type=int,                                         help="number of training events")
args = argParser.parse_args()

# import the toy model
import toy_models as models
model = getattr( models, args.model )

# directory for plots
plot_directory = os.path.join( user_plot_directory, args.plot_directory, args.model )

if not os.path.isdir(plot_directory):
    os.makedirs( plot_directory )

# initiate plot
Plot.setDefaults()

xmin = 1
xmax = 5
support    = [xmin, xmax]
pdf        = model.Pow1D
#parameters = [(2,),(1.5,),(1.25,)]
#theta_ref  = [0,0]
#plot_theta_values = [(0,1), (1,0), (1,1)]
#theta      = [0,0]

parameters = [(2,),(1.25,)]
#parameters = [(2,),(2,)]
theta_ref  = [0]
plot_theta_values = [(0.,) ,(-.5,), (-.25,), (.25,), (.5,), ]
#plot_theta_values = [(.25,), (.5,),(.75,), (1,), ]
theta      = [0]

mixturePDF = model.QuadraticMixturePDF( pdf, parameters, support )

training_features = mixturePDF.getEvents(  args.nTraining, theta_ref = theta_ref)
training_weights  = mixturePDF.getWeights( training_features, theta = theta, theta_ref = theta_ref)

# Text on the plots
def drawObjects( offset=0 ):
    tex1 = ROOT.TLatex()
    tex1.SetNDC()
    tex1.SetTextSize(0.05)
    tex1.SetTextAlign(11) # align right

    tex2 = ROOT.TLatex()
    tex2.SetNDC()
    tex2.SetTextSize(0.04)
    tex2.SetTextAlign(11) # align right

    line1 = ( 0.15+offset, 0.95, "Boosted Info Trees" )
    #line2 = ( 0.68, 0.95, "%i fb^{-1} (13 TeV)" )

    return [ tex1.DrawLatex(*line1) ]#, tex2.DrawLatex(*line2) ]


# Plot a 1D histogram
def plot1DHist( plot, plot_directory, yRange=(0.3,"auto"), ratio={'yRange':(0.1,1.9)}, legend=(0.2,0.7,0.9,0.9), lumi=137, plotLog=True, histModifications=[], titleOffset=0 ):

    for log in [True, False] if plotLog else [False]:

        if yRange[0] == 0 and log:
            yRange = list(yRange)
            yRange[0] = 0.0003
            yRange = tuple(yRange)

        # Add subdirectory for lin/log plots
        plot_directory_ = os.path.join( plot_directory, "log" if log else "lin" )

        plotting.draw( plot,
                       plot_directory = plot_directory_,
                       logX = False, logY = log, sorting = False,
                       yRange = yRange,
                       ratio = ratio,
#                       drawObjects = drawObjects( lumi, offset=titleOffset ),
                       legend = legend,
                       histModifications = histModifications,
                       copyIndexPHP = True,
                       )

###############
## Plot Model #
###############

bsm_colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan]

# Let's plot the model so that Niki sees the hypothesis.
Nbins = 80
h_BSM = {theta:ROOT.TH1F("h_BSM", "h_BSM", Nbins, xmin, xmax) for theta in plot_theta_values}
h_theory = {theta:ROOT.TH1F("h_theory", "h_theory", Nbins, xmin, xmax) for theta in plot_theta_values}

for i_theta, theta in enumerate(plot_theta_values):
    weights_BSM = mixturePDF.getWeights( training_features, theta = theta, theta_ref = theta_ref, only_weights = True) 
    for i in range(args.nTraining):
        h_BSM[theta].Fill( training_features[i], weights_BSM[i] )

    h_BSM[theta].style = styles.lineStyle( bsm_colors[i_theta], width=2, dashed=False )
    h_BSM[theta].Scale(Nbins/(xmax-xmin)/h_BSM[theta].Integral())
    #h_BSM[theta].Scale(1./args.nTraining)
    #h_BSM[theta].Scale(1./mixturePDF.sigma(theta))
    h_BSM[theta].legendText = "sampled #theta=%s"%str(theta)
    
    for i_bin in range(1,h_theory[theta].GetNbinsX()+1):
        h_theory[theta].SetBinContent( i_bin, mixturePDF.eval( theta, [h_theory[theta].GetBinLowEdge(i_bin)] ))
        h_theory[theta].style = styles.lineStyle( bsm_colors[i_theta], width=2, dashed=True )
        #h_theory[theta].Scale(1./h_theory[theta].Integral())
        h_theory[theta].legendText = "p(x|#theta=%s)"%str(theta)


# Plot of model and theory 
histos = []
for theta in plot_theta_values:
    histos.append([h_theory[theta]])
    histos.append([h_BSM[theta]])

plot   = Plot.fromHisto( "model",  histos, texX="x", texY="a.u." )

# Plot Style
histModifications      = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

ratioHistModifications = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
ratio                  = None #{'yRange':(0.51,1.49), 'texY':"BSM/SM", "histModifications":ratioHistModifications}
legend                 = [(0.2,0.74,0.8,0.88),2]
yRange                 = (0.00003, "auto")
#    yRange                 = (0, "auto")

plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, histModifications=histModifications )

#######################
### Plot some regions #
#######################
#
#region_colors = [ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan]
#
#regions = [
#    {'legendText':"1#leq x < 1.05", "bin":(1,1.05)},
#    {'legendText':"2#leq x < 2.5", "bin":(2,2.5)},
#    {'legendText':"3#leq x < 5",    "bin":(3,5)},
#
#   ]
#
## Let's plot the model so that Niki sees the hypothesis.
#h_r = [ROOT.TH1F("h_BSM", "h_BSM", 50, -3, 2) for region in regions]
#
#for i_region, region in enumerate(regions):
#    events = training_features[(training_features[:,0]>region["bin"][0])&(training_features[:,0]<region["bin"][1]),:] 
#    histo = h_r[i_region]
#    print( "%i events for region %i"%(len(events), i_region))
#    for i_bin in range(1,histo.GetNbinsX()+1):
#        weights= mixturePDF.getWeights( events, theta = (histo.GetBinLowEdge(i_bin),), theta_ref = theta_ref, only_weights = True)
#        #print (histo.GetBinLowEdge(i_bin),), weights.sum()
#        histo.SetBinContent( i_bin, weights.sum() ) 
#    histo.style = styles.lineStyle( region_colors[i_region], width=2 )
#    histo.legendText = region['legendText'] 
#
## Plot of hypothesis
#histos = [ [h]  for h in h_r] 
#plot   = Plot.fromHisto( "regions",  histos, texX="#theta", texY="a.u." )
#
## Plot Style
#histModifications      = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
#histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
#histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
#histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
#histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]
#
#ratioHistModifications = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
#ratio                  = None #{'yRange':(0.51,1.49), 'texY':"BSM/SM", "histModifications":ratioHistModifications}
#legend                 = (0.2,0.74,0.6,0.88)
#yRange                 = (0.00003, "auto")
##    yRange                 = (0, "auto")
#
#plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, histModifications=histModifications )
#
##############
##############

# Boosting
n_trees       = 50
max_depth     = 2
learning_rate = 0.20
min_size      = 50
n_plot        = 10

test_features = mixturePDF.getEvents( args.nTraining, theta_ref = theta_ref)
test_weights  = mixturePDF.getWeights( test_features, theta = theta, theta_ref = theta_ref)

bits = {}
for derivative in training_weights.keys():
    if derivative == tuple(): continue

    filename = "bit_mixture_derivative_%i"%derivative if len(derivative)==1 else "bit_mixture_derivative_%i_%i"%derivative
    try:
        print ("Loading %s for %r"%( filename, derivative))
        bits[derivative] = BoostedInformationTree.load(filename+'.pkl')
    except IOError:
        time1 = time.time()
        print ("Learning %s"%( str(derivative)))
        bits[derivative]= BoostedInformationTree(
                training_features     = training_features,
                training_weights      = training_weights[tuple()],
                training_diff_weights = training_weights[derivative],
                learning_rate         = learning_rate,
                n_trees               = n_trees,
                max_depth             = max_depth,
                min_size              = min_size,
                split_method          = 'vectorized_split_and_weight_sums',
                weights_update_method = 'vectorized',
                calibrated            = False,
                    )
        bits[derivative].boost()
        bits[derivative].save(filename+'.pkl')
        print ("Written %s"%( filename ))

        time2 = time.time()
        boosting_time = time2 - time1
        print ("Boosting time: %.2f seconds" % boosting_time)

        # plot loss
        test_scores     = bits[derivative].vectorized_predict(test_features)
        training_scores = bits[derivative].vectorized_predict(test_features)
        max_score = max(test_scores)
        min_score = min(test_scores)

        test_FIs            = np.zeros(n_trees)
        training_FIs        = np.zeros(n_trees)

        for i in range(args.nTraining):
            test_scores     = bits[derivative].predict( test_features[i],     summed = False)
            training_scores = bits[derivative].predict( training_features[i], summed = False)

            test_score  = sum( test_scores )
            train_score = sum( training_scores )

            # compute test and training FI evolution during training
            test_FIs     += test_weights[derivative][i]*test_scores
            training_FIs += training_weights[derivative][i]*training_scores

        training_FI_histo     = ROOT.TH1D("trainFI", "trainFI",          n_trees, 1, n_trees+1 )
        test_FI_histo         = ROOT.TH1D("testFI",  "testFI",           n_trees, 1, n_trees+1 )

        for i_tree in range(n_trees):
            test_FI_histo    .SetBinContent( i_tree+1, -sum(test_FIs[:i_tree]) )
            training_FI_histo.SetBinContent( i_tree+1, -sum(training_FIs[:i_tree]) )

        # Histo style
        test_FI_histo    .style = styles.lineStyle( ROOT.kBlue, width=2 )
        training_FI_histo.style = styles.lineStyle( ROOT.kRed, width=2 )
        test_FI_histo    .legendText = "Test"
        training_FI_histo.legendText = "Training"

        minY   = 0.01 * min( test_FI_histo.GetBinContent(test_FI_histo.GetMaximumBin()), training_FI_histo.GetBinContent(training_FI_histo.GetMaximumBin()))
        maxY   = 1.5  * max( test_FI_histo.GetBinContent(test_FI_histo.GetMaximumBin()), training_FI_histo.GetBinContent(training_FI_histo.GetMaximumBin()))

        histos = [ [test_FI_histo], [training_FI_histo] ]
        plot   = Plot.fromHisto( filename+"_evolution", histos, texX="b", texY="L(D,b)" )

        # Plot Style
        histModifications      = []
        histModifications      += [ lambda h: h.GetYaxis().SetTitleOffset(1.4) ]
        histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
        histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
        histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
        histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

        ratioHistModifications = []
        ratio                  = None
        legend                 = (0.6,0.75,0.9,0.88)
        yRange                 = "auto" #( minY, maxY )

        plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, plotLog=False, titleOffset=0.08, histModifications=histModifications )

histos_weights = [] 
histos_scores  = [] 
Nbins             = 8
derivative_colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan]
i_derivative      = 0

derivatives = sorted(test_weights.keys())[1:]

for derivative in derivatives:
    h_w = ROOT.TH1F("h_w", "h_w", 8, xmin, xmax)
    h_w.style      = styles.lineStyle(derivative_colors[i_derivative], dashed = True)
    h_w.legendText = "weights der: %s"%(str(derivative))
    histos_weights.append(h_w)
    h_s = ROOT.TH1F("h_w", "h_w", 8, xmin, xmax)
    h_s.style      = styles.lineStyle(derivative_colors[i_derivative])
    h_s.legendText = "scores der: %s"%(str(derivative))
    histos_scores.append(h_s)
    i_derivative   += 1

for i_bin in range(Nbins):
    bin = (xmin + i_bin*(xmax-xmin)/float(Nbins), xmin + (i_bin+1)*(xmax-xmin)/float(Nbins))
    print ("Working at bin", bin)
    mask   = (test_features[:,0]>bin[0])&(test_features[:,0]<bin[1]) 
    events = test_features[mask,:]

    for i_derivative, derivative in enumerate(derivatives):

        print ("bin", bin, "derivative", derivative)

        event_weights = test_weights[derivative][mask]
        event_scores = bits[derivative].vectorized_predict(events)
        print         
        print event_weights
        print event_scores
        
        histos_weights[i_derivative].SetBinContent( i_bin+1, event_weights.sum()/len(event_weights) )
        histos_scores [i_derivative].SetBinContent( i_bin+1, event_scores.sum()/len(event_scores) )

histos = [[h] for h in histos_weights] + [[h] for h in histos_scores]
plot   = Plot.fromHisto( "coefficients", histos, texX="x", texY="coefficient" )

# Plot Style
histModifications      = []
histModifications      += [ lambda h: h.GetYaxis().SetTitleOffset(1.4) ]
histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

ratioHistModifications = []
ratio                  = None
legend                 = (0.6,0.75,0.9,0.88)
yRange                 = "auto" #( minY, maxY )

plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, plotLog=False, titleOffset=0.08, histModifications=histModifications )

################################
### Plot likelihood comparison #
################################
#
#bsm_colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan]
#test_features = mixturePDF.getEvents( args.nTraining, theta_ref = theta_ref)
#test_weights  = mixturePDF.getWeights( test_features, theta = theta, theta_ref = theta_ref)
#
#Nbins = 80
#l_reco    = {theta:ROOT.TH1F("l_reco", "l_reco", Nbins, xmin, xmax) for theta in plot_theta_values}
#l_theory  = {theta:ROOT.TH1F("l_theory", "l_theory", Nbins, xmin, xmax) for theta in plot_theta_values}
#
#for i_theta, theta in enumerate(plot_theta_values):
#    for i in range(args.nTraining)[:100]:
#        lr_theory = mixturePDF.eval(theta, test_features[i]) / mixturePDF.eval(theta_ref, test_features[i])
##        l_theory[theta].Fill( test_features[i], lr_theory )
##        lr_reco   = 1 + (theta[0]-theta_ref[0])*bits[(0,)].predict(test_features[i]) + 0.5*(theta[0]-theta_ref[0])**2*bits[(0,0)].predict(test_features[i])
##        l_reco[theta].Fill( test_features[i], lr_reco )
#
#        #print "theta", theta, "theta_ref",theta_ref, "features", test_features[i], "lr_theory", lr_theory, "lr_reco", lr_reco
#        print "theta", theta, "theta_ref",theta_ref, "features", test_features[i], "bits0 / 00",bits[(0,)].predict(test_features[i]), bits[(0,0)].predict(test_features[i]), "w",test_weights[()][i], test_weights[(0,)][i],test_weights[(0,0)][i]
#         assert False, "fix bug"
##    l_theory[theta].style = styles.lineStyle( bsm_colors[i_theta], width=2, dashed=False )
##    #l_theory[theta].Scale(Nbins/(xmax-xmin)/l_BSM[theta].Integral())
##    #l_theory[theta].Scale(1./args.nTraining)
##    #l_theory[theta].Scale(1./mixturePDF.sigma(theta))
##    l_theory[theta].legendText = "sampled #theta=%s"%str(theta)
##    
##    for i_bin in range(1,l_theory[theta].GetNbinsX()+1):
##        l_theory[theta].SetBinContent( i_bin, mixturePDF.eval( theta, [l_theory[theta].GetBinLowEdge(i_bin)] ))
##        l_theory[theta].style = styles.lineStyle( bsm_colors[i_theta], width=2, dashed=True )
##        #l_theory[theta].Scale(1./l_theory[theta].Integral())
##        l_theory[theta].legendText = "p(x|#theta=%s)"%str(theta)
#
#
## Plot of model and theory 
#histos = []
#for theta in plot_theta_values:
#    histos.append([l_theory[theta]])
#    histos.append([l_reco[theta]])
#
#plot   = Plot.fromHisto( "model",  histos, texX="x", texY="a.u." )
#
## Plot Style
#histModifications      = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
#histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
#histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
#histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
#histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]
#
#ratioHistModifications = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
#ratio                  = None #{'yRange':(0.51,1.49), 'texY':"BSM/SM", "histModifications":ratioHistModifications}
#legend                 = [(0.2,0.74,0.8,0.88),2]
#yRange                 = (0.00003, "auto")
##    yRange                 = (0, "auto")
#
#plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, histModifications=histModifications )
