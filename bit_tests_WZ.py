#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
from math import log, exp, sin, cos, sqrt, pi
import copy

# RootTools
from RootTools.core.standard   import *

# Analysis
import Analysis.Tools.syncer

# BIT
from BoostedInformationTree import BoostedInformationTree

# User
from user import plot_directory as user_plot_directory

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="BIT_v1",                                                        help="plot sub-directory")
argParser.add_argument("--nTraining",          action="store",      default=500000,        type=int,                                         help="number of training events")
args = argParser.parse_args()

# import the toy model
import toy_models.WZ as model

# directory for plots
plot_directory = os.path.join( user_plot_directory, args.plot_directory, "WZ" )

if not os.path.isdir(plot_directory):
    os.makedirs( plot_directory )

# initiate plot
Plot.setDefaults()

training_features, training_extra = model.getEvents(args.nTraining)
training_weights = model.getWeights(training_features, eft=model.default_eft_parameters)

efts = map(lambda e:model.make_eft(**e), [ {}, {'cW':0.2}, {'cW':.4}, {'cW':0.2, 'c3PQ':0.2}, {'c3PQ':0.2}, {'c3PQ':0.4} ] )


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

Nbins = 50
funcs   = {'s':sqrt,             'Theta':cos,                                                          'thetaW':cos,                'thetaZ':cos}
binning = {'s':[Nbins,600,3000], 'Theta':[Nbins,-1,1],  'phiW':[Nbins,0,2*pi], 'phiZ':[Nbins,0,2*pi],  'thetaW':[Nbins,-1,1],       'thetaZ':[Nbins,-1,1],       'lep_w_charge':[3,-1,2]}
nice_name={'s':"#sqrt{s}",       'Theta':"cos(#Theta)", 'phiW':"#phi_{W}",     'phiZ':"#phi_{Z}",      'thetaW':"cos(#theta_{W})",  'thetaZ':"cos(#theta_{Z})",  'lep_w_charge':"charge(l_{W})"}

colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan, ROOT.kRed]

h={}
for i_eft, eft in enumerate(efts):
    weights         = model.getWeights(training_features, eft)
    name = ''
    name= '_'.join( [ (wc+'_%3.2f'%eft[wc]).replace('.','p').replace('-','m') for wc in model.wilson_coefficients if eft.has_key(wc) ])
    if name=='': name='SM'
    h[name] = {}
    eft['name']=name
    for i_feature, feature in enumerate(model.feature_names):
        h[name][feature] = ROOT.TH1F(name+'_'+feature, name+'_'+feature, *binning[feature] )

    for i_event, event in enumerate(training_features):
        for i_feature, feature in enumerate(model.feature_names):
            func = funcs[feature] if funcs.has_key(feature) else lambda x:x
            h[name][feature].Fill(func(event[i_feature]), weights[()][i_event])
            h[name][feature].style = styles.lineStyle( colors[i_eft], width=2, dashed=False )
            h[name][feature].legendText = name

for i_feature, feature in enumerate(model.feature_names):
    histos = [[h[eft['name']][feature]] for eft in efts]
    plot   = Plot.fromHisto( feature,  histos, texX=nice_name[feature], texY="a.u." )

    for log in [True, False]:

        # Add subdirectory for lin/log plots
        plot_directory_ = os.path.join( plot_directory, "log" if log else "lin" )

        plotting.draw( plot,
                       plot_directory = plot_directory_,
                       logX = False, logY = log, sorting = False,
                       yRange = "auto",
                       ratio = None,
#                       drawObjects = drawObjects( lumi, offset=titleOffset ),
                        legend=(0.2,0.7,0.9,0.9),
                       #histModifications = histModifications,
                       copyIndexPHP = True,
                       )

# Boosting
n_trees       = 150
max_depth     = 4
learning_rate = 0.20
min_size      = 50

test_features,_ = model.getEvents( args.nTraining)
test_weights    = model.getWeights( test_features, eft=model.default_eft_parameters)

bits = {}
for derivative in training_weights.keys():
    if derivative == tuple(): continue

    filename = "bit_WZ_derivative_%s"%("_".join(derivative))
    try:
        print ("Loading %s for %r"%( filename, derivative))
        bits[derivative] = BoostedInformationTree.load(filename+'.pkl')
    except IOError:
        time1 = time.time()
        print ("Learning %s"%("_".join(derivative)))
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

assert False, ""
##############
# Plot Score #
##############


bit_predictions  = { key:bits[key].vectorized_predict(test_features) for key in  training_weights.keys() if key!=tuple() }

h = {}
param = 'c3PQ'
for i_feature_name, feature_name in enumerate(model.feature_names):
    h[feature_name] = {}
    h[feature_name]["predicted"] = ROOT.TH1D( feature_name, feature_name, *binning[feature_name] )
    h[feature_name]["predicted"].legendText = "predicted" 
    h[feature_name]["predicted"].style      = styles.lineStyle( ROOT.kRed, width = 2)

    h[feature_name]["simulated"] = ROOT.TH1D( feature_name, feature_name, *binning[feature_name] )
    h[feature_name]["simulated"].legendText = "simulated" 
    h[feature_name]["simulated"].style      = styles.lineStyle( ROOT.kBlue, width = 2)

    feature_values = test_features[:,i_feature_name]

    for i_feature, feature in enumerate(feature_values):
        func = funcs[feature_name] if funcs.has_key(feature_name) else lambda x:x
        h[feature_name]["simulated"].Fill(func(feature), test_weights[(param,)][i_feature]) 
        h[feature_name]["predicted"].Fill(func(feature), test_weights[()][i_feature]*bit_predictions[(param,)][i_feature]) 

    histos = [ [h[feature_name]["predicted"]], [h[feature_name]["simulated"]] ]
    plot   = Plot.fromHisto( "score_%s_%s"%(feature_name,param), histos, texX=nice_name[feature_name], texY="score" )

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
    yRange                 = "auto" #(-10,10) 

    plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, plotLog=False, titleOffset=0.08, histModifications=histModifications )

h_truth     = ROOT.TH1F("h_truth", "h_truth", 40, -.03, .05)
h_pred      = ROOT.TH1F("h_pred", "h_pred",   40, -.03, .05)
h_pred_lin  = ROOT.TH1F("h_pred_lin", "h_pred_lin",   40, -.03, .05)
h_truth.style = styles.lineStyle( ROOT.kBlack, width=2, dashed=False )
h_pred .style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
h_pred_lin .style = styles.lineStyle( ROOT.kGreen, width=2, dashed=True )
h_truth.legendText = "ext. log(L(x|#theta)/L(x|0))"
h_pred.legendText  = "pred. log-likelihood ratio"
h_pred.legendText  = "predicted (quadratic)"
h_pred_lin.legendText  = "predicted (linear)"

def compute_weights(weights, theta, theta_ref, param):
   return weights[()] \
    + np.array( [ (theta-theta_ref)*weights[(param,)]]).sum(axis=0)\
    + np.array( [ 0.5*(theta-theta_ref)*(theta-theta_ref)*weights[(param, param)]] ).sum(axis=0)

def predict_weight_ratio(bit_predictions, theta, theta_ref, param):
    lin       = np.array( [ (theta-theta_ref)*bit_predictions[(param,)] ] ).sum(axis=0)
    quadratic = np.array( [ 0.5*(theta-theta_ref)*(theta-theta_ref)*bit_predictions[(param, param)] ] ).sum(axis=0)
    return 1.+lin, 1.+lin+quadratic

for i_plot_theta, plot_theta in enumerate(np.arange(-.03,.05,.002)):
    w1 = compute_weights( test_weights, plot_theta, 0, param)
    w0 = compute_weights( test_weights, 0,0, param)
 
    ext_Delta_NLL      = w1 - w0 - w0*np.log(w1/w0)
 
    lin, quad = np.log(predict_weight_ratio(bit_predictions, plot_theta, 0, param))
    ext_Delta_NLL_pred     = w1 - w0 - w0*quad
    ext_Delta_NLL_pred_lin = w1 - w0 - w0*lin

    i_bin = h_truth.FindBin(plot_theta)

    ext_Delta_NLL_sum = ext_Delta_NLL.sum()
    ext_Delta_NLL_pred_sum = ext_Delta_NLL_pred.sum()
    ext_Delta_NLL_pred_lin_sum = ext_Delta_NLL_pred_lin.sum()

    if ext_Delta_NLL_sum<float('inf'):
        h_truth.SetBinContent( i_bin, ext_Delta_NLL_sum )
    if ext_Delta_NLL_pred_sum<float('inf'):
        h_pred.SetBinContent( i_bin, ext_Delta_NLL_pred_sum )
    if ext_Delta_NLL_pred_lin_sum<float('inf'):
        h_pred_lin.SetBinContent( i_bin, ext_Delta_NLL_pred_lin_sum )

    print plot_theta, "true", ext_Delta_NLL.sum(), "pred", ext_Delta_NLL_pred.sum(), "lin", ext_Delta_NLL_pred_lin.sum()

plot   = Plot.fromHisto( "likelihood",  [[h_truth], [h_pred], [h_pred_lin]], texX="#theta", texY="LLR" )

histModifications      = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

ratioHistModifications = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
ratio                  = None #{'yRange':(0.51,1.49), 'texY':"BSM/SM", "histModifications":ratioHistModifications}
legend                 = [(0.2,0.74,0.8,0.88),1]
yRange                 = "auto"
plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, histModifications=histModifications )


#########################################
### Plot unbinned likelihood comparison #
#########################################
#
#bsm_colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan]
#
#bit_predictions  = { key:bits[key].vectorized_predict(test_features) for key in  training_weights.keys() if key!=tuple() }
#
#def compute_weights(weights, theta, theta_ref):
#   return weights[()] \
#    + np.array( [ (theta[i]-theta_ref[i])*weights[(i,)] for i in range(len(theta)) ] ).sum(axis=0)\
#    + np.array( [ 0.5*(theta[i]-theta_ref[i])*(theta[j]-theta_ref[j])*weights[(i,j)] for i in range(len(theta)) for j in range(len(theta)) ] ).sum(axis=0)
#
#def predict_weight_ratio(bit_predictions, theta, theta_ref):
#   return 1.+ \
#    + np.array( [ (theta[i]-theta_ref[i])*bit_predictions[(i,)] for i in range(len(theta)) ] ).sum(axis=0)\
#    + np.array( [ 0.5*(theta[i]-theta_ref[i])*(theta[j]-theta_ref[j])*bit_predictions[(i,j)] for i in range(len(theta)) for j in range(len(theta)) ] ).sum(axis=0)
#
#for i_plot_theta, plot_theta in enumerate(np.arange(-1,1,.05).reshape(-1,1)):
#    w1 = compute_weights( test_weights, plot_theta, theta_ref ) 
#    w0 = compute_weights( test_weights, theta_ref , theta_ref ) 
#
#    ext_Delta_NLL      = w1 - w0 - w0*np.log(w1/w0)
#    ext_Delta_NLL_pred = w1 - w0 - w0*np.log(predict_weight_ratio(bit_predictions, plot_theta, theta_ref))
#    print plot_theta, "true", ext_Delta_NLL.sum(), "pred", ext_Delta_NLL_pred.sum()
#
