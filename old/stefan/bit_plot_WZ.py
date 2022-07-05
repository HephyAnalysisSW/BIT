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
from ROOT import TLegend 

# RootTools
from RootTools.core.standard   import *

# Analysis
import Analysis.Tools.syncer as syncer

# BIT
from BoostedInformationTree import BoostedInformationTree

# User
from user import plot_directory as user_plot_directory

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="BIT_v1",                                                        help="plot sub-directory")
argParser.add_argument("--nTraining",          action="store",      default=500000,        type=int,                                         help="number of training events")
argParser.add_argument("--derivative",         action="store",      default=None,          nargs="*",                                        help="Maximum number of splits in node split")
argParser.add_argument('--overwrite',          action='store_true', help="Overwrite output?")
args = argParser.parse_args()

# import the toy model
import toy_models.WZ as model

# directory for plots
plot_directory = os.path.join( user_plot_directory, args.plot_directory, "WZ" )

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

# initiate plot
Plot.setDefaults()

training_features, training_extra = model.getEvents(args.nTraining)
training_weights = model.getWeights(training_features, eft=model.default_eft_parameters)

if args.derivative is not None:
    for key in training_weights.keys():
        if key not in ( tuple(), tuple(args.derivative) ):
            del training_weights[key]

print "nEvents: %i Weights: %s" %( len(training_features), [ k for k in training_weights.keys() if k!=tuple()] )

efts = map(lambda e:model.make_eft(**e), [ {}, {'cW':0.2}, {'cW':.4}, {'cW':0.2, 'c3PQ':0.2}, {'c3PQ':0.2}, {'c3PQ':0.4} ] )

## prints for checking polynomial coeffs
#for i_event in range(1):
#    for i_eft, eft in enumerate(efts):
#        w = model.getWeights( training_features[i_event:i_event+1], eft=eft )
#        w_pred = training_weights[()][i_event]
#        # linear term
#        for coeff in model.wilson_coefficients:
#            w_pred += (eft[coeff]-model.default_eft_parameters[coeff])*training_weights[(coeff,)][i_event]
#        print "eft:",eft, "getWeights:", w[()], "at base point:", training_weights[()][i_event], "derivative %s"%coeff,training_weights[(coeff,)][i_event], "prediction:", w_pred

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

colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan, ROOT.kRed]

h   ={}
h_rw={}
for i_eft, eft in enumerate(efts):
    weights         = model.getWeights(training_features, eft)

    if i_eft == 0:
        weights_sm = weights
        eft_sm     = eft
    name = ''
    name= '_'.join( [ (wc+'_%3.2f'%eft[wc]).replace('.','p').replace('-','m') for wc in model.wilson_coefficients if eft.has_key(wc) ])
    if name=='': name='SM'
    h[name] = {}
    h_rw[name] = {}
    eft['name']=name
    for i_feature, feature in enumerate(model.feature_names):
        h[name][feature] = ROOT.TH1F(name+'_'+feature, name+'_'+feature, *model.plot_options[feature]['binning'] )
        h_rw[name][feature] = ROOT.TH1F(name+'_'+feature, name+'_'+feature, *model.plot_options[feature]['binning'] )

    for i_event, event in enumerate(training_features):
        for i_feature, feature in enumerate(model.feature_names):

            h[name][feature].Fill(event[i_feature], weights[()][i_event])
            h[name][feature].style = styles.lineStyle( colors[i_eft], width=2, dashed=False )
            h[name][feature].legendText = name

            reweight = weights_sm[()][i_event]\
                        + (eft['cW']-eft_sm['cW'])*weights_sm[('cW',)][i_event]\
                        + (eft['c3PQ']-eft_sm['c3PQ'])*weights_sm[('c3PQ',)][i_event]\
                        + 0.5*(eft['cW']-eft_sm['cW'])**2*weights_sm[('cW','cW')][i_event]\
                        + 0.5*(eft['c3PQ']-eft_sm['c3PQ'])**2*weights_sm[('c3PQ','c3PQ')][i_event]\
                        + (eft['cW']-eft_sm['cW'])*(eft['c3PQ']-eft_sm['c3PQ'])*weights_sm[('c3PQ','cW')][i_event]

            h_rw[name][feature].Fill(event[i_feature], reweight)
            h_rw[name][feature].style = styles.lineStyle( colors[i_eft], width=2, dashed=False )
            h_rw[name][feature].legendText = name

for i_feature, feature in enumerate(model.feature_names):
    histos = [[h[eft['name']][feature]] for eft in efts]
    plot   = Plot.fromHisto( feature,  histos, texX=model.plot_options[feature]['tex'], texY="a.u." )
    histos_rw = [[h_rw[eft['name']][feature]] for eft in efts]
    plot_rw= Plot.fromHisto( feature+'_rw',  histos_rw, texX=model.plot_options[feature]['tex'], texY="a.u." )

    for log in [True, False]:

        # Add subdirectory for lin/log plots
        plot_directory_ = os.path.join( plot_directory, "log" if log else "lin" )
        for p in [plot, plot_rw]:
            plotting.draw( p,
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

    filename = os.path.join(plot_directory, "bit_WZ_derivative_%s"%("_".join(derivative)))
    try:
        print ("Loading %s for %r"%( filename, derivative))
        bits[derivative] = BoostedInformationTree.load(filename+'.pkl')
    except IOError:
        args.overwrite = True

    if args.overwrite:
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
                global_score_subtraction = False,
                
                    )
        bits[derivative].boost(debug=True)
        bits[derivative].save(filename+'.pkl')
        print ("Written %s"%( filename ))

        time2 = time.time()
        boosting_time = time2 - time1
        print ("Boosting time: %.2f seconds" % boosting_time)

        from debug import make_debug_plots
        make_debug_plots( bits[derivative],
                          training_features, training_weights[tuple()],
                          training_weights[derivative],
                          test_features,
                          test_weights[tuple()],
                          test_weights[derivative],
                          os.path.join(plot_directory, ('_'.join(derivative))),
                          mva_variables = [ model.plot_options[name]['tex'] for name in model.feature_names ] )#model.mva_variables)

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

##############
# Plot Score #
##############

bit_predictions_test  = { key:bits[key].vectorized_predict(test_features) for key in  training_weights.keys() if key!=tuple() }
bit_predictions_train = { key:bits[key].vectorized_predict(training_features) for key in  training_weights.keys() if key!=tuple() }

h = {}
#for param in model.wilson_coefficients:
for derivative in training_weights.keys():
    if derivative == tuple(): continue
    for i_feature_name, feature_name in enumerate(model.feature_names):
        h[feature_name] = {}
        h[feature_name]["predicted_train"] = ROOT.TH1D( feature_name, feature_name, *model.plot_options[feature_name]['binning'] )
        h[feature_name]["predicted_train"].legendText = "predicted (train)" 
        h[feature_name]["predicted_train"].style      = styles.lineStyle( ROOT.kRed, width = 2)

        h[feature_name]["simulated_train"] = ROOT.TH1D( feature_name, feature_name, *model.plot_options[feature_name]['binning'] )
        h[feature_name]["simulated_train"].legendText = "simulated (train)" 
        h[feature_name]["simulated_train"].style      = styles.lineStyle( ROOT.kBlue, width = 2)

        h[feature_name]["predicted_test"] = ROOT.TH1D( feature_name, feature_name, *model.plot_options[feature_name]['binning'] )
        h[feature_name]["predicted_test"].legendText = "predicted (test)" 
        h[feature_name]["predicted_test"].style      = styles.lineStyle( ROOT.kRed, width = 2, dashed=True)

        h[feature_name]["simulated_test"] = ROOT.TH1D( feature_name, feature_name, *model.plot_options[feature_name]['binning'] )
        h[feature_name]["simulated_test"].legendText = "simulated (test)" 
        h[feature_name]["simulated_test"].style      = styles.lineStyle( ROOT.kBlue, width = 2, dashed = True)

        feature_values = test_features[:,i_feature_name]
        for i_feature, feature in enumerate(feature_values):
            h[feature_name]["simulated_test"].Fill(feature, test_weights[derivative][i_feature]) 
            h[feature_name]["predicted_test"].Fill(feature, test_weights[()][i_feature]*bit_predictions_test[derivative][i_feature]) 

        feature_values = training_features[:,i_feature_name]
        for i_feature, feature in enumerate(feature_values):
            h[feature_name]["simulated_train"].Fill(feature, training_weights[derivative][i_feature]) 
            h[feature_name]["predicted_train"].Fill(feature, training_weights[()][i_feature]*bit_predictions_train[derivative][i_feature]) 

        histos = [ [h[feature_name]["predicted_train"]], [h[feature_name]["simulated_train"]], [h[feature_name]["predicted_test"]], [h[feature_name]["simulated_test"]] ]
        plot   = Plot.fromHisto( "score_%s_%s"%(feature_name,"_".join(list(derivative))), histos, texX=model.plot_options[feature_name]['tex'], texY="score" )

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

directory = os.path.join(plot_directory, 'LLR')

if not os.path.exists(directory):
    try:
        os.makedirs(directory)
    except IOError:
        pass

c1 = ROOT.TCanvas()
f = open(os.path.join(directory,'Polynomials.txt'),'w')
f.write("Polynomials in cW:")
f.write("\n")
for c3 in [-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0]:
    h_LLR_true = ROOT.TH1D("h_LLR_true","LLR Weights Summed",40,-1.0,1.0)
    h_LLR_true_event = ROOT.TH1D("h_LLR_true_event","LLR Eventwise",40,-1.0,1.0)
    h_LLR_pred_lin = ROOT.TH1D("h_LLR_pred_lin","LLR Weights Summed",40,-1.0,1.0)
    h_LLR_pred_lin_event = ROOT.TH1D("h_LLR_pred_lin_event","LLR Eventwise",40,-1.0,1.0)
    h_LLR_pred_quad = ROOT.TH1D("h_LLR_pred_quad","LLR Weights Summed",40,-1.0,1.0)
    h_LLR_pred_quad_event = ROOT.TH1D("h_LLR_pred_quad_event","LLR Eventwise",40,-1.0,1.0)
    vec_lin_pred = []
    vec_quad_pred = []
    vec_lin_pred_event = []
    vec_quad_pred_event = []
    vec_theta = []
    for i_plot_theta, plot_theta in enumerate(np.arange(-1.0,1.0,0.05)):
        w0_sum = 0
        w1_true_sum = 0
        w1_pred_quad_sum = 0
        w1_pred_lin_sum = 0
        LLR_true_event = 0
        LLR_true_capped_event = 0
        LLR_pred_lin_event = 0
        LLR_pred_quad_event = 0
        for j in range(test_features[:,0].size):
                w1_true_inter = test_weights[()][j] + plot_theta*test_weights[('cW',)][j] + 0.5 * plot_theta**2 * test_weights[('cW', 'cW')][j] + c3*test_weights[('c3PQ',)][j] + 0.5 * c3**2 * test_weights[('c3PQ', 'c3PQ')][j] +  plot_theta*c3*test_weights[('c3PQ', 'cW')][j] #mixed term => factor 2 compensates 1/2
                w0_inter = test_weights[()][j]
                div = np.divide(w1_true_inter,w0_inter,out = np.zeros_like(w1_true_inter),where=w0_inter!=0)
                LLR_true_event += w1_true_inter - w0_inter -w0_inter*np.log(div,out = np.zeros_like(div),where=div>0)
                #if div > 0: #only add not nan logs (div > 0)
                #    LLR_true_event += w1_true_inter - w0_inter -w0_inter*np.log(div)
                w1_true_sum += w1_true_inter
                w0_sum += w0_inter
                w1_pred_quad_inter = w0_inter + plot_theta * bit_predictions_test[('cW',)][j]*w0_inter + 0.5*plot_theta**2 * bit_predictions_test[('cW', 'cW')][j]*w0_inter + c3 * bit_predictions_test[('c3PQ',)][j]*w0_inter + 0.5*c3**2 * bit_predictions_test[('c3PQ', 'c3PQ')][j]*w0_inter + plot_theta*c3 * bit_predictions_test[('c3PQ', 'cW')][j]*w0_inter
                w1_pred_quad_sum += w1_pred_quad_inter
                div = np.divide(w1_pred_quad_inter,w0_inter,out = np.zeros_like(w1_pred_quad_inter),where=w0_inter!=0)
                LLR_pred_quad_event += w1_pred_quad_inter - w0_inter -w0_inter*np.log(div,out = np.zeros_like(div),where=div>0)
                #if div > 0:
                #    LLR_pred_quad_event += w1_pred_quad_inter - w0_inter -w0_inter*np.log(div)
                w1_pred_lin_inter = w0_inter + plot_theta * bit_predictions_test[('cW',)][j]*w0_inter + c3 * bit_predictions_test[('c3PQ',)][j]*w0_inter
                w1_pred_lin_sum += w1_pred_lin_inter
                div = np.divide(w1_pred_lin_inter,w0_inter,out = np.zeros_like(w1_pred_lin_inter),where=w0_inter!=0)
                LLR_pred_lin_event += w1_pred_lin_inter - w0_inter -w0_inter*np.log(div,out = np.zeros_like(div),where=div>0)
                #if div > 0:
                #    LLR_pred_lin_event += w1_pred_lin_inter - w0_inter -w0_inter*np.log(div)
        div = w1_true_sum/w0_sum
        LLR_true = w1_true_sum - w0_sum - w0_sum*np.log(div,out = np.zeros_like(div),where=div>0)
        div = w1_pred_quad_sum/w0_sum
        LLR_pred_quad = w1_pred_quad_sum - w0_sum - w0_sum*np.log(div,out = np.zeros_like(div),where=div>0)
        div = w1_pred_lin_sum/w0_sum
        LLR_pred_lin = w1_pred_lin_sum - w0_sum - w0_sum*np.log(div,out = np.zeros_like(div),where=div>0)
        h_LLR_true.Fill(plot_theta+0.01,LLR_true)
        h_LLR_true_event.Fill(plot_theta+0.01,LLR_true_event)
        h_LLR_pred_quad_event.Fill(plot_theta+0.01,LLR_pred_quad_event)
        h_LLR_pred_lin_event.Fill(plot_theta+0.01,LLR_pred_lin_event)
        h_LLR_pred_quad.Fill(plot_theta+0.01,LLR_pred_quad)
        h_LLR_pred_lin.Fill(plot_theta+0.01,LLR_pred_lin)
        vec_lin_pred.append(LLR_pred_lin)
        vec_quad_pred.append(LLR_pred_quad)
        vec_lin_pred_event.append(LLR_pred_lin_event)
        vec_quad_pred_event.append(LLR_pred_quad_event)
        vec_theta.append(plot_theta)
    lin_fit = np.polyfit(vec_theta,vec_lin_pred,1)
    quad_fit = np.polyfit(vec_theta,vec_quad_pred,2)
    lin_event_fit = np.polyfit(vec_theta,vec_lin_pred_event,1)
    quad_event_fit = np.polyfit(vec_theta,vec_quad_pred_event,2)
    f.write("cW Polynom - c3PQ_%s"% ('_'.join([str(round(c3,2))])))
    f.write("\n")
    f.write("Linear Polynom: ")
    f.write("x0: "+str(lin_fit[0])+" ,x1: "+str(lin_fit[1]))
    f.write("\n")
    f.write("Quadratic Polynom: ")
    f.write("x0: "+str(quad_fit[0])+" ,x1: "+str(quad_fit[1])+" ,x2: "+str(quad_fit[2]))
    f.write("\n")
    f.write("Eventwise cW Polynom - c3PQ_%s"% ('_'.join([str(round(c3,2))])))
    f.write("\n")
    f.write("Linear Polynom: ")
    f.write("x0: "+str(lin_event_fit[0])+" ,x1: "+str(lin_event_fit[1]))
    f.write("\n")
    f.write("Quadratic Polynom: ")
    f.write("x0: "+str(quad_event_fit[0])+" ,x1: "+str(quad_event_fit[1])+" ,x2: "+str(quad_event_fit[2]))
    f.write("\n")

    print "########################################################"
    h_LLR_true.SetTitle("LLR Weights Summed;cW;LLR")
    h_LLR_true.GetYaxis().SetTitleOffset(1.4)
    h_LLR_true.SetLineColor(ROOT.kRed)
    h_LLR_true_event.SetLineColor(ROOT.kRed)
    h_LLR_pred_lin.SetLineColor(ROOT.kGreen)
    h_LLR_pred_lin_event.SetLineColor(ROOT.kGreen)
    h_LLR_pred_quad.SetLineColor(ROOT.kBlue)
    h_LLR_pred_quad_event.SetLineColor(ROOT.kBlue)
    legend = TLegend(0.65,0.82,0.95,0.92)
    legend.SetFillStyle(0)
    legend.SetShadowColor(ROOT.kWhite)
    legend.SetBorderSize(0)
    legend.AddEntry(h_LLR_true,"LLR true","l")
    legend.AddEntry(h_LLR_pred_lin,"LLR linear prediction","l")
    legend.AddEntry(h_LLR_pred_quad,"LLR quadratic prediction","l")
    #h_LLR_true.Draw("hist")
    #h_LLR_pred_lin.Draw("histsame")
    #h_LLR_pred_quad.Draw("histsame")
    #legend.Draw()
    #filename = "LLR_cW_c3PQ_%s"% ('_'.join([str(round(c3,2))])) + ".png"
    #c1.Print(os.path.join(directory,filename))
    h_LLR_true.legendText = "LLR true"
    h_LLR_pred_lin.legendText = "LLR linear prediction"
    h_LLR_pred_quad.legendText = "LLR quadratic prediction"
    histos = [[h_LLR_true],[h_LLR_pred_lin],[h_LLR_pred_quad]]
    plot   = Plot.fromHisto( "LLR_cW_c3PQ_%s"% ('_'.join([str(round(c3,2))])),  histos, texX = "cW", texY = "LLR" )
    plotting.draw(plot,
                    plot_directory = directory,
                    yRange = 'auto', logY = False, logX = False,
                    )

    h_LLR_true_event.SetTitle("LLR Eventwise cW;cW;LLR")
    legend = TLegend(0.65,0.82,0.95,0.92)
    legend.SetFillStyle(0)
    legend.SetShadowColor(ROOT.kWhite)
    legend.SetBorderSize(0)
    legend.AddEntry(h_LLR_true_event,"LLR true","l")
    legend.AddEntry(h_LLR_pred_lin_event,"LLR linear prediction","l")
    legend.AddEntry(h_LLR_pred_quad_event,"LLR quadratic prediction","l")
    #h_LLR_true_event.Draw("hist")
    #h_LLR_pred_lin_event.Draw("histsame")
    #h_LLR_pred_quad_event.Draw("histsame")
    #legend.Draw()
    #filename = "LLR_cW_eventwise_c3PQ_%s"% ('_'.join([str(round(c3,2))])) + ".png" 
    #c1.Print(os.path.join(directory,filename))

    h_LLR_true_event.legendText = "LLR true"
    h_LLR_pred_lin_event.legendText = "LLR linear prediction"
    h_LLR_pred_quad_event.legendText = "LLR quadratic prediction"
    histos = [[h_LLR_true_event],[h_LLR_pred_lin_event],[h_LLR_pred_quad_event]]
    plot   = Plot.fromHisto( "LLR_cW_eventwise_c3PQ_%s"% ('_'.join([str(round(c3,2))])),  histos, texX = "cW", texY = "LLR" )
    plotting.draw(plot,
                    plot_directory = directory,
                    yRange = 'auto', logY = False, logX = False,
                    )

for cW in [-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0]:
    h_LLR_true = ROOT.TH1D("h_LLR_true","LLR Weights Summed",40,-1.0,1.0)
    h_LLR_true_event = ROOT.TH1D("h_LLR_true_event","LLR Eventwise",40,-1.0,1.0)
    h_LLR_pred_lin = ROOT.TH1D("h_LLR_pred_lin","LLR Weights Summed",40,-1.0,1.0)
    h_LLR_pred_lin_event = ROOT.TH1D("h_LLR_pred_lin_event","LLR Eventwise",40,-1.0,1.0)
    h_LLR_pred_quad = ROOT.TH1D("h_LLR_pred_quad","LLR Weights Summed",40,-1.0,1.0)
    h_LLR_pred_quad_event = ROOT.TH1D("h_LLR_pred_quad_event","LLR Eventwise",40,-1.0,1.0)
    vec_lin_pred = []
    vec_quad_pred = []
    vec_lin_pred_event = []
    vec_quad_pred_event = []
    vec_theta = []
    for i_plot_theta, plot_theta in enumerate(np.arange(-1.0,1.0,0.05)):
        w0_sum = 0
        w1_true_sum = 0
        w1_pred_quad_sum = 0
        w1_pred_lin_sum = 0
        LLR_true_event = 0
        LLR_true_capped_event = 0
        LLR_pred_lin_event = 0
        LLR_pred_quad_event = 0
        for j in range(test_features[:,0].size):
                w1_true_inter = test_weights[()][j] + plot_theta*test_weights[('c3PQ',)][j] + 0.5 * (plot_theta**2) * test_weights[('c3PQ', 'c3PQ')][j] + cW*test_weights[('cW',)][j] + 0.5 * (cW**2) * test_weights[('cW', 'cW')][j]  + plot_theta*cW*test_weights[('c3PQ', 'cW')][j]
                w0_inter = test_weights[()][j]
                div = np.divide(w1_true_inter,w0_inter,out = np.zeros_like(w1_true_inter),where=w0_inter!=0)
                LLR_true_event += w1_true_inter - w0_inter -w0_inter*np.log(div,out = np.zeros_like(div),where=div>0)
                #if div > 0:
                #    LLR_true_event += w1_true_inter - w0_inter -w0_inter*np.log(div)
                w1_true_sum += w1_true_inter
                w0_sum += w0_inter
                w1_pred_quad_inter = w0_inter + plot_theta * bit_predictions_test[('c3PQ',)][j]*w0_inter + 0.5*(plot_theta**2) * bit_predictions_test[('c3PQ', 'c3PQ')][j]*w0_inter + cW * bit_predictions_test[('cW',)][j]*w0_inter + 0.5*(cW**2) * bit_predictions_test[('cW', 'cW')][j]*w0_inter  + plot_theta * cW* bit_predictions_test[('c3PQ', 'cW')][j]*w0_inter
                w1_pred_quad_sum += w1_pred_quad_inter
                div = np.divide(w1_pred_quad_inter,w0_inter,out = np.zeros_like(w1_pred_quad_inter),where=w0_inter!=0)
                LLR_pred_quad_event += w1_pred_quad_inter - w0_inter -w0_inter*np.log(div,out = np.zeros_like(div),where=div>0)
                #if div > 0:
                #    LLR_pred_quad_event += w1_pred_quad_inter - w0_inter -w0_inter*np.log(div)
                w1_pred_lin_inter = w0_inter + plot_theta * bit_predictions_test[('c3PQ',)][j]*w0_inter + cW * bit_predictions_test[('cW',)][j]*w0_inter
                w1_pred_lin_sum += w1_pred_lin_inter
                div = np.divide(w1_pred_lin_inter,w0_inter,out = np.zeros_like(w1_pred_lin_inter),where=w0_inter!=0)
                LLR_pred_lin_event += w1_pred_lin_inter - w0_inter -w0_inter*np.log(div,out = np.zeros_like(div),where=div>0)
                #if div > 0:
                #    LLR_pred_lin_event += w1_pred_lin_inter - w0_inter -w0_inter*np.log(div)
        div = w1_true_sum/w0_sum
        LLR_true = w1_true_sum - w0_sum - w0_sum*np.log(div,out = np.zeros_like(div),where=div>0)
        div = w1_pred_quad_sum/w0_sum
        LLR_pred_quad = w1_pred_quad_sum - w0_sum - w0_sum*np.log(div,out = np.zeros_like(div),where=div>0)
        div = w1_pred_lin_sum/w0_sum
        LLR_pred_lin = w1_pred_lin_sum - w0_sum - w0_sum*np.log(div,out = np.zeros_like(div),where=div>0)
        h_LLR_true.Fill(plot_theta+0.01,LLR_true)
        h_LLR_true_event.Fill(plot_theta+0.01,LLR_true_event)
        h_LLR_pred_quad_event.Fill(plot_theta+0.01,LLR_pred_quad_event)
        h_LLR_pred_lin_event.Fill(plot_theta+0.01,LLR_pred_lin_event)
        h_LLR_pred_quad.Fill(plot_theta+0.01,LLR_pred_quad)
        h_LLR_pred_lin.Fill(plot_theta+0.01,LLR_pred_lin)
        vec_lin_pred.append(LLR_pred_lin)
        vec_quad_pred.append(LLR_pred_quad)
        vec_lin_pred_event.append(LLR_pred_lin_event)
        vec_quad_pred_event.append(LLR_pred_quad_event)
        vec_theta.append(plot_theta)
    lin_fit = np.polyfit(vec_theta,vec_lin_pred,1)
    quad_fit = np.polyfit(vec_theta,vec_quad_pred,2)
    lin_event_fit = np.polyfit(vec_theta,vec_lin_pred_event,1)
    quad_event_fit = np.polyfit(vec_theta,vec_quad_pred_event,2)
    f.write("c3PQ Polynom - cW_%s"% ('_'.join([str(round(cW,2))])))
    f.write("\n")
    f.write("Linear Polynom: ")
    f.write("x0: "+str(lin_fit[0])+" ,x1: "+str(lin_fit[1]))
    f.write("\n")
    f.write("Quadratic Polynom: ")
    f.write("x0: "+str(quad_fit[0])+" ,x1: "+str(quad_fit[1])+" ,x2: "+str(quad_fit[2]))
    f.write("\n")
    f.write("Eventwise c3PQ Polynom - cW_%s"% ('_'.join([str(round(cW,2))])))
    f.write("\n")
    f.write("Linear Polynom: ")
    f.write("x0: "+str(lin_event_fit[0])+" ,x1: "+str(lin_event_fit[1]))
    f.write("\n")
    f.write("Quadratic Polynom: ")
    f.write("x0: "+str(quad_event_fit[0])+" ,x1: "+str(quad_event_fit[1])+" ,x2: "+str(quad_event_fit[2]))
    f.write("\n")

    print "########################################################"
    h_LLR_true.SetTitle("LLR Weights Summed;c3PQ;LLR")
    h_LLR_true.GetYaxis().SetTitleOffset(1.4)
    h_LLR_true.SetLineColor(ROOT.kRed)
    h_LLR_true_event.SetLineColor(ROOT.kRed)
    h_LLR_pred_lin.SetLineColor(ROOT.kGreen)
    h_LLR_pred_lin_event.SetLineColor(ROOT.kGreen)
    h_LLR_pred_quad.SetLineColor(ROOT.kBlue)
    h_LLR_pred_quad_event.SetLineColor(ROOT.kBlue)
    legend = TLegend(0.65,0.82,0.95,0.92)
    legend.SetFillStyle(0)
    legend.SetShadowColor(ROOT.kWhite)
    legend.SetBorderSize(0)
    legend.AddEntry(h_LLR_true,"LLR true","l")
    legend.AddEntry(h_LLR_pred_lin,"LLR linear prediction","l")
    legend.AddEntry(h_LLR_pred_quad,"LLR quadratic prediction","l")
    #h_LLR_true.Draw("hist")
    #h_LLR_pred_lin.Draw("histsame")
    #h_LLR_pred_quad.Draw("histsame")
    #legend.Draw()
    #filename = "LLR_c3PQ_cW_%s"% ('_'.join([str(round(cW,2))])) + ".png"
    #c1.Print(os.path.join(directory,filename))
    h_LLR_true.legendText = "LLR true"
    h_LLR_pred_lin.legendText = "LLR linear prediction"
    h_LLR_pred_quad.legendText = "LLR quadratic prediction"
    histos = [[h_LLR_true],[h_LLR_pred_lin],[h_LLR_pred_quad]]
    plot   = Plot.fromHisto( "LLR_c3PQ_cW_%s"% ('_'.join([str(round(cW,2))])),  histos, texX = "c3PQ", texY = "LLR" )
    plotting.draw(plot,
                    plot_directory = directory,
                    yRange = 'auto', logY = False, logX = False,
                    )

    h_LLR_true_event.SetTitle("LLR Eventwise;c3PQ;LLR")
    legend = TLegend(0.65,0.82,0.95,0.92)
    legend.SetFillStyle(0)
    legend.SetShadowColor(ROOT.kWhite)
    legend.SetBorderSize(0)
    legend.AddEntry(h_LLR_true_event,"LLR true","l")
    legend.AddEntry(h_LLR_pred_lin_event,"LLR linear prediction","l")
    legend.AddEntry(h_LLR_pred_quad_event,"LLR quadratic prediction","l")
    #h_LLR_true_event.Draw("hist")
    #h_LLR_pred_lin_event.Draw("histsame")
    #h_LLR_pred_quad_event.Draw("histsame")
    #legend.Draw()
    #filename = "LLR_c3PQ_eventwise_cW_%s"% ('_'.join([str(round(cW,2))])) + ".png"
    #c1.Print(os.path.join(directory,filename))

    h_LLR_true_event.legendText = "LLR true"
    h_LLR_pred_lin_event.legendText = "LLR linear prediction"
    h_LLR_pred_quad_event.legendText = "LLR quadratic prediction"
    histos = [[h_LLR_true_event],[h_LLR_pred_lin_event],[h_LLR_pred_quad_event]]
    plot   = Plot.fromHisto( "LLR_c3PQ_eventwise_cW_%s"% ('_'.join([str(round(cW,2))])),  histos, texX = "c3PQ", texY = "LLR" )
    plotting.draw(plot,
                    plot_directory = directory,
                    yRange = 'auto', logY = False, logX = False,
                    )
#print test_weights
f.close()




#h_truth     = ROOT.TH1F("h_truth", "h_truth", 40, -.03, .05)
#h_pred      = ROOT.TH1F("h_pred", "h_pred",   40, -.03, .05)
#h_pred_lin  = ROOT.TH1F("h_pred_lin", "h_pred_lin",   40, -.03, .05)
#h_truth.style = styles.lineStyle( ROOT.kBlack, width=2, dashed=False )
#h_pred .style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
#h_pred_lin .style = styles.lineStyle( ROOT.kGreen, width=2, dashed=True )
#h_truth.legendText = "ext. log(L(x|#theta)/L(x|0))"
#h_pred.legendText  = "pred. log-likelihood ratio"
#h_pred.legendText  = "predicted (quadratic)"
#h_pred_lin.legendText  = "predicted (linear)"
#
#def compute_weights(weights, theta, theta_ref, param):
#   return weights[()] \
#    + np.array( [ (theta-theta_ref)*weights[(param,)]]).sum(axis=0)\
#    + np.array( [ 0.5*(theta-theta_ref)*(theta-theta_ref)*weights[(param, param)]] ).sum(axis=0)
#
#def predict_weight_ratio(bit_predictions, theta, theta_ref, param):
#    lin       = np.array( [ (theta-theta_ref)*bit_predictions[(param,)] ] ).sum(axis=0)
#    quadratic = np.array( [ 0.5*(theta-theta_ref)*(theta-theta_ref)*bit_predictions[(param, param)] ] ).sum(axis=0)
#    return 1.+lin, 1.+lin+quadratic
#
#param = 'cW'
#for i_plot_theta, plot_theta in enumerate(np.arange(-.03,.05,.002)):
#    w1 = compute_weights( test_weights, plot_theta, 0, param)
#    w0 = compute_weights( test_weights, 0,0, param)
# 
#    ext_Delta_NLL      = w1 - w0 - w0*np.log(w1/w0)
# 
#    lin, quad = np.log(predict_weight_ratio(bit_predictions, plot_theta, 0, param))
#    ext_Delta_NLL_pred     = w1 - w0 - w0*quad
#    ext_Delta_NLL_pred_lin = w1 - w0 - w0*lin
#
#    i_bin = h_truth.FindBin(plot_theta)
#
#    ext_Delta_NLL_sum = ext_Delta_NLL.sum()
#    ext_Delta_NLL_pred_sum = ext_Delta_NLL_pred.sum()
#    ext_Delta_NLL_pred_lin_sum = ext_Delta_NLL_pred_lin.sum()
#
#    if ext_Delta_NLL_sum<float('inf'):
#        h_truth.SetBinContent( i_bin, ext_Delta_NLL_sum )
#    if ext_Delta_NLL_pred_sum<float('inf'):
#        h_pred.SetBinContent( i_bin, ext_Delta_NLL_pred_sum )
#    if ext_Delta_NLL_pred_lin_sum<float('inf'):
#        h_pred_lin.SetBinContent( i_bin, ext_Delta_NLL_pred_lin_sum )
#
#    print plot_theta, "true", ext_Delta_NLL.sum(), "pred", ext_Delta_NLL_pred.sum(), "lin", ext_Delta_NLL_pred_lin.sum()
#
#plot   = Plot.fromHisto( "likelihood",  [[h_truth], [h_pred], [h_pred_lin]], texX="#theta", texY="LLR" )
#
#histModifications      = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
#histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
#histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
#histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
#histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]
#
#ratioHistModifications = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
#ratio                  = None #{'yRange':(0.51,1.49), 'texY':"BSM/SM", "histModifications":ratioHistModifications}
#legend                 = [(0.2,0.74,0.8,0.88),1]
#yRange                 = "auto"
#plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, histModifications=histModifications )
#
#
