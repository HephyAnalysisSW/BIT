#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
from math import log, exp, sqrt
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
argParser.add_argument("--model",              action="store",      default="exponential", type=str,   choices=allModels,                    help="import model")
argParser.add_argument("--theta",              action="store",      default=0.001,         type=float,                                       help="theta value for model")
argParser.add_argument("--nTraining",          action="store",      default=100000,        type=int,                                         help="number of training events")
argParser.add_argument("--luminosity",         action="store",      default=137,           type=int,                                         help="luminosity value, currently only for plot label")
argParser.add_argument("--lowPtThresh",        action="store",      default=50,            type=int,                                         help="low pt threshold")
argParser.add_argument("--highPtThresh",       action="store",      default=200,           type=int,                                         help="high pt threshold")
argParser.add_argument("--fraction_alpha",     action="store",      default=1,             type=float,                                       help="Move information to this fraction of the events (for overtraining study)")
argParser.add_argument("--max_relative_score_uncertainty", action="store", default=0,      type=float,                                       help="Maximum allowed relative score uncertainty in each node")
args = argParser.parse_args()

# import the toy model
import toy_models as models
model = getattr( models, args.model )

# Produce training data set
training_features, training_weights, training_diff_weights = model.get_dataset( args.nTraining )

def weight_up(fraction_alpha, diff_weights):
    bool_arr = np.zeros(len(diff_weights), dtype=bool)
    bool_arr[:int(round(fraction_alpha*len(diff_weights)))] = True
    random.shuffle(bool_arr)
    diff_weights[bool_arr]  = diff_weights[bool_arr]/args.fraction_alpha
    diff_weights[~bool_arr] = 0.

# Move the whole information into a fraction 'fraction_alpha' of events
model_postfix = ''
if args.fraction_alpha>0 and args.fraction_alpha<1:
    weight_up( args.fraction_alpha, training_diff_weights)
    model_postfix += "_alpha%4.4f"%args.fraction_alpha

if args.max_relative_score_uncertainty>0:
   model_postfix += "_maxJN%4.4f"%args.max_relative_score_uncertainty

# directory for plots
plot_directory = os.path.join( user_plot_directory, args.plot_directory, model.id_string + model_postfix )

if not os.path.isdir(plot_directory):
    os.makedirs( plot_directory )

# initiate plot
Plot.setDefaults()


# Text on the plots
def drawObjects( lumi, offset=0 ):
    tex1 = ROOT.TLatex()
    tex1.SetNDC()
    tex1.SetTextSize(0.05)
    tex1.SetTextAlign(11) # align right

    tex2 = ROOT.TLatex()
    tex2.SetNDC()
    tex2.SetTextSize(0.04)
    tex2.SetTextAlign(11) # align right

    line1 = ( 0.15+offset, 0.95, "Boosted Info Trees" )
    line2 = ( 0.68, 0.95, "%i fb^{-1} (13 TeV)"%lumi )

    return [ tex1.DrawLatex(*line1), tex2.DrawLatex(*line2) ]


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

##############
# Plot Model #
##############

# Let's plot the model so that Niki sees the hypothesis.
h_SM  = ROOT.TH1F("h_SM",  "h_SM",  40, model.xmin, model.xmax)
h_BSM = ROOT.TH1F("h_BSM", "h_BSM", 40, model.xmin, model.xmax)

for i in range(args.nTraining):
    h_SM.Fill ( training_features[i], training_weights[i] )
    h_BSM.Fill( training_features[i], training_weights[i]+args.theta*training_diff_weights[i] )

# Histo style
h_SM.style = styles.lineStyle( ROOT.kRed, width=2 )
h_SM.Scale(1./h_SM.Integral())

h_BSM.style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
h_BSM.Scale(1./h_BSM.Integral())

if args.model == "exponential" or "mixture":
    h_SM.legendText = "#theta_{0}=%.2f"%model.theta0
    h_BSM.legendText = "#theta_{0}+#Delta#theta, #Delta#theta=%.3f"%args.theta
else:
    h_SM.legendText = "#theta_{0}=%i"%model.theta0
    h_BSM.legendText = "#theta_{0}+#Delta#theta, #Delta#theta=%.1f"%args.theta


# Plot of hypothesis
histos = [ [h_SM], [h_BSM] ]
plot   = Plot.fromHisto( "model",  histos, texX="x", texY="a.u." )

# Plot Style
histModifications      = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

ratioHistModifications = [] #lambda h: h.GetYaxis().SetTitleOffset(2.2) ]
ratio                  = None #{'yRange':(0.51,1.49), 'texY':"BSM/SM", "histModifications":ratioHistModifications}
legend                 = (0.2,0.74,0.6,0.88)
if args.model == "power_law":
    yRange                 = (0.00003, "auto")
else:
    yRange                 = (0, "auto")

plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, lumi=args.luminosity, histModifications=histModifications )

##############
##############


# Boosting
time1 = time.time()

# BIT config
n_trees       = 50
max_depth     = 2
learning_rate = 0.20
min_size      = 100
n_plot        = 5

bit= BoostedInformationTree(
        training_features     = training_features,
        training_weights      = training_weights,
        training_diff_weights = training_diff_weights,
        learning_rate         = learning_rate,
        n_trees               = n_trees,
        max_depth             = max_depth,
        min_size              = min_size,
        split_method          = 'vectorized_split_and_weight_sums' if args.max_relative_score_uncertainty==0 else 'iterative_split_and_weight_sums',
        weights_update_method = 'vectorized',
        calibrated            = False,
        max_relative_score_uncertainty= args.max_relative_score_uncertainty
            )

bit.boost()

#bit.save('tmp.pkl')
#bit = BoostedInformationTree.load('tmp.pkl')

time2 = time.time()
boosting_time = time2 - time1
print ("Boosting time: %.2f seconds" % boosting_time)

# Testing
test_features, test_weights, test_diff_weights = model.get_dataset( args.nTraining )
if args.fraction_alpha>0 and args.fraction_alpha<1:
    weight_up( args.fraction_alpha, test_diff_weights)

training_profile     = ROOT.TProfile("trainP", "trainP",         8, model.xmin, model.xmax)
test_profile         = ROOT.TProfile("testP",  "testP",          8, model.xmin, model.xmax)
training_BSM_profile = ROOT.TProfile("BSM_trainP", "BSM_trainP", 8, model.xmin, model.xmax)
test_BSM_profile     = ROOT.TProfile("BSM_testP",  "BSM_testP",  8, model.xmin, model.xmax)

training     = ROOT.TH1D("train", "train",          8, model.min_score_theory, model.max_score_theory )
test         = ROOT.TH1D("test",  "test",           8, model.min_score_theory, model.max_score_theory )
training_BSM = ROOT.TH1D("BSM_train", "BSM_train",  8, model.min_score_theory, model.max_score_theory )
test_BSM     = ROOT.TH1D("BSM_test",  "BSM_test",   8, model.min_score_theory, model.max_score_theory )

training_FI_histo     = ROOT.TH1D("trainFI", "trainFI",          n_trees, 1, n_trees+1 )
test_FI_histo         = ROOT.TH1D("testFI",  "testFI",           n_trees, 1, n_trees+1 )

test_FIs            = np.zeros(n_trees)
training_FIs        = np.zeros(n_trees)
test_FIs_lowPt      = np.zeros(n_trees)
training_FIs_lowPt  = np.zeros(n_trees)
test_FIs_highPt     = np.zeros(n_trees)
training_FIs_highPt = np.zeros(n_trees)


# Weight variance measure
VM            = ROOT.TH1D("VM", "VM", 20, model.xmin, model.xmax)
VMwp_den      = ROOT.TH1D("VMwp_den", "VMwp_den", 20, model.xmin, model.xmax)
VM_den        = ROOT.TH1D("VM_den", "VM_den", 20, model.xmin, model.xmax)

VM_test       = ROOT.TH1D("VM_test", "VM_test", 20, model.xmin, model.xmax)
VM_testwp_den = ROOT.TH1D("VM_testwp_den", "VM_testwp_den", 20, model.xmin, model.xmax)
VM_test_den   = ROOT.TH1D("VM_test_den", "VM_test_den", 20, model.xmin, model.xmax)

for i in range(args.nTraining):
    test_scores     = bit.predict( test_features[i],     summed = False)
    training_scores = bit.predict( training_features[i], summed = False)

    test_score  = sum( test_scores )
    train_score = sum( training_scores )

    test_profile    .Fill(      test_features[i][0],     test_score,   test_weights[i] )
    training_profile.Fill(      training_features[i][0], train_score,  training_weights[i] )
    test_BSM_profile    .Fill(  test_features[i][0],     test_score,   test_weights[i]+args.theta*test_diff_weights[i] )
    training_BSM_profile.Fill(  training_features[i][0], train_score,  training_weights[i]+args.theta*training_diff_weights[i] )
    test    .Fill(       test_score, test_weights[i])
    training.Fill(       train_score, training_weights[i])
    test_BSM    .Fill(   test_score,  test_weights[i]+args.theta*test_diff_weights[i])
    training_BSM.Fill(   train_score, training_weights[i]+args.theta*training_diff_weights[i])

    # compute test and training FI evolution during training
    test_FIs     += test_diff_weights[i]*test_scores
    training_FIs += training_diff_weights[i]*training_scores
    if test_features[i][0]<args.lowPtThresh:
        test_FIs_lowPt          += test_diff_weights[i]*test_scores
    if training_features[i][0]<args.lowPtThresh:
        training_FIs_lowPt      += training_diff_weights[i]*training_scores
    if test_features[i][0]>args.highPtThresh:
        test_FIs_highPt         += test_diff_weights[i]*test_scores
    if training_features[i][0]>args.highPtThresh:
        training_FIs_highPt     += training_diff_weights[i]*training_scores

    # compute VM = sum(w'^2/w) / (sum(w')^2/sum(w) ) - 1
    if training_weights[i]!=0:
        VM.Fill(      training_features[i][0], training_diff_weights[i]**2/training_weights[i]) 
        VMwp_den.Fill(training_features[i][0], training_diff_weights[i]) 
        VM_den.Fill(  training_features[i][0], training_weights[i]) 
    if test_weights[i]!=0:
        VM_test.Fill(       test_features[i][0], test_diff_weights[i]**2/test_weights[i]) 
        VM_testwp_den.Fill( test_features[i][0], test_diff_weights[i]) 
        VM_test_den.Fill(   test_features[i][0], test_weights[i]) 

for i_bin in range(1, 1+VM.GetNbinsX() ):
    if VM_den.GetBinContent(i_bin)!=0 and VMwp_den.GetBinContent(i_bin):
        #VM.SetBinContent( i_bin, -1+VM.GetBinContent(i_bin)/(VMwp_den.GetBinContent(i_bin)**2/VM_den.GetBinContent(i_bin)))
        VM.SetBinContent( i_bin, (VMwp_den.GetBinContent(i_bin)**2/VM_den.GetBinContent(i_bin))/VM.GetBinContent(i_bin))
    else:
        VM.SetBinContent( i_bin, 0.)
for i_bin in range(1, 1+VM_test.GetNbinsX() ):
    if VM_test_den.GetBinContent(i_bin)!=0 and VM_testwp_den.GetBinContent(i_bin):
        VM_test.SetBinContent( i_bin, (VM_testwp_den.GetBinContent(i_bin)**2/VM_test_den.GetBinContent(i_bin))/VM_test.GetBinContent(i_bin))
        #print i_bin, VM_test.GetBinContent(i_bin), VM_testwp_den.GetBinContent(i_bin)**2, VM_test_den.GetBinContent(i_bin), VM_test.GetBinContent(i_bin)/(VM_testwp_den.GetBinContent(i_bin)**2/VM_test_den.GetBinContent(i_bin)) 
        #print i_bin, VM.GetBinContent(i_bin), VM_test.GetBinContent(i_bin)
    else:
        VM_test.SetBinContent( i_bin, 0.)

########################
# Plot Weight Variance #
########################

# Histo style
VM.style = styles.lineStyle( ROOT.kRed, width=2, dashed=True )
VM_test.style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
VM.legendText      = "Training (#alpha_f=%4.4f)"%args.fraction_alpha
VM_test.legendText = "Test"

#VMwp_den.style = styles.lineStyle( ROOT.kRed, width=2, dashed=True )
#VM_testwp_den.style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
#VMwp_den.legendText      = "Training (#alpha_f=%4.4f)"%args.fraction_alpha
#VM_testwp_den.legendText = "Test"
#
#VM_den.style = styles.lineStyle( ROOT.kRed, width=2, dashed=True )
#VM_test_den.style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
#VM_den.legendText      = "Training (#alpha_f=%4.4f)"%args.fraction_alpha
#VM_test_den.legendText = "Test"

# Plot of hypothesis

# Plot Style
histModifications      = [ lambda h: h.GetYaxis().SetTitleOffset(1.4) ]
histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

ratioHistModifications = []
ratio                  = None
legend                 = (0.5,0.65,0.9,0.9)

plot = Plot.fromHisto( "VM",  [ [VM], [VM_test] ], texX=model.texX, texY="VM" )
plot1DHist( plot,  plot_directory, yRange="auto", ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=False, histModifications=histModifications )
#plot = Plot.fromHisto( "VM_den",  [ [VM_den], [VM_test_den] ])
#plot1DHist( plot, plot_directory, yRange="auto", ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=False, histModifications=histModifications )
#plot1DHist( plot, plot_directory, yRange="auto", ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=True, histModifications=histModifications )
#plot = Plot.fromHisto( "VMwp_den",  [ [VMwp_den], [VM_testwp_den] ])
#plot1DHist( plot, plot_directory, yRange="auto", ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=False, histModifications=histModifications )
#plot1DHist( plot, plot_directory, yRange="auto", ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=True, histModifications=histModifications )

######################################
# Stat uncertainty of score function #
######################################

n_bins = 20

JN        = ROOT.TH1D("JN", "JN", n_bins, model.xmin, model.xmax)
JN_test   = ROOT.TH1D("JN_test", "JN_test", n_bins, model.xmin, model.xmax)

JN_rel        = ROOT.TH1D("JN_rel", "JN_rel", n_bins, model.xmin, model.xmax)
JN_rel_test   = ROOT.TH1D("JN_rel_test", "JN_rel_test", n_bins, model.xmin, model.xmax)

digi   = np.digitize(training_features[:,0], np.arange(model.xmin, model.xmax, (model.xmax-model.xmin)/float(n_bins)))
for n_bin in range(1,n_bins+1):
    events = digi==n_bin
    n      = np.sum(events)
    all_but_one      = ( -training_diff_weights[events]+np.sum(training_diff_weights[events]) ) / (-training_weights[events]+np.sum(training_weights[events]) )
    mean_all_but_one = np.mean(all_but_one)
    uncertainty = sqrt( (n-1.)/n*( np.sum( ( all_but_one-mean_all_but_one)**2 ) )  )
    JN.SetBinContent( n_bin, uncertainty )
    JN_rel.SetBinContent( n_bin, uncertainty/np.sum(training_weights[events]) )
digi   = np.digitize(test_features[:,0], np.arange(model.xmin, model.xmax, (model.xmax-model.xmin)/float(n_bins)))
for n_bin in range(1,n_bins+1):
    events = digi==n_bin
    n      = np.sum(events)
    all_but_one      = ( -test_diff_weights[events]+np.sum(test_diff_weights[events]) ) / (-test_weights[events]+np.sum(test_weights[events]) )
    mean_all_but_one = np.mean(all_but_one)
    uncertainty = sqrt( (n-1.)/n*( np.sum( ( all_but_one-mean_all_but_one)**2 ) )  )
    JN_test.SetBinContent( n_bin, uncertainty )
    JN_rel_test.SetBinContent( n_bin, uncertainty/np.sum(test_weights[events]) )

JN.style = styles.lineStyle( ROOT.kRed, width=2, dashed=True )
JN_test.style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
JN.legendText      = "Training (#alpha_f=%4.4f)"%args.fraction_alpha
JN_test.legendText = "Test"
JN_rel.style = styles.lineStyle( ROOT.kRed, width=2, dashed=True )
JN_rel_test.style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
JN_rel.legendText      = "Training (#alpha_f=%4.4f)"%args.fraction_alpha
JN_rel_test.legendText = "Test"

# Plot Style
histModifications      = [ lambda h: h.GetYaxis().SetTitleOffset(1.4) ]
histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

ratioHistModifications = []
ratio                  = None
legend                 = (0.5,0.65,0.9,0.9)

plot = Plot.fromHisto( "JN",  [ [JN], [JN_test] ], texX=model.texX, texY="JN" )
plot1DHist( plot,  plot_directory, yRange="auto", ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=False, histModifications=histModifications )
plot1DHist( plot,  plot_directory, yRange="auto", ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=True, histModifications=histModifications )
plot = Plot.fromHisto( "JN_rel",  [ [JN_rel], [JN_rel_test] ], texX=model.texX, texY="JN_rel" )
plot1DHist( plot,  plot_directory, yRange="auto", ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=False, histModifications=histModifications )
plot1DHist( plot,  plot_directory, yRange="auto", ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=True, histModifications=histModifications )

######################
# Plot Score Profile #
######################

# Histo style
training_profile.style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
test_profile    .style = styles.lineStyle( ROOT.kBlue, width=2 )
training_BSM_profile.style = styles.lineStyle( ROOT.kRed, width=2, dashed=True )
test_BSM_profile    .style = styles.lineStyle( ROOT.kRed, width=2 )

training_profile.legendText = "Training (#theta_{0})"
test_profile    .legendText = "Test (#theta_{0})"
training_BSM_profile.legendText = "Training (#theta_{0}+#Delta#theta)"
test_BSM_profile    .legendText = "Test (#theta_{0}+#Delta#theta)"

# Plot of hypothesis
histos = [ [training_profile], [test_profile], [training_BSM_profile], [test_BSM_profile] ]
plot   = Plot.fromHisto( "score_profile_validation_profile",  histos, texX=model.texX, texY="F(x)" )

# Plot Style
histModifications      = [ lambda h: h.GetYaxis().SetTitleOffset(1.4) ]
histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

ratioHistModifications = []
ratio                  = None
legend                 = (0.55,0.6,0.9,0.9)
minY                   = model.min_score_theory
maxY                   = model.max_score_theory
yRange                 = (minY, maxY)

#plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=False, histModifications=histModifications )

##############
##############


#########################
# Plot Score Validation #
#########################

# Histo style
training.style = styles.lineStyle( ROOT.kBlue, width=2, dashed=True )
test    .style = styles.lineStyle( ROOT.kBlue, width=2 )
training_BSM.style = styles.lineStyle( ROOT.kRed, width=2, dashed=True )
test_BSM    .style = styles.lineStyle( ROOT.kRed, width=2 )

training.legendText = "Training (#theta_{0})"
test    .legendText = "Test (#theta_{0})"
training_BSM.legendText = "Training (#theta_{0}+#Delta#theta)"
test_BSM    .legendText = "Test (#theta_{0}+#Delta#theta)"

test.Scale(1./training.Integral())
test_BSM.Scale(1./training_BSM.Integral())
training.Scale(1./training.Integral())
training_BSM.Scale(1./training_BSM.Integral())

# Plot of hypothesis
histos = [ [training], [test], [training_BSM], [test_BSM] ]

plot   = Plot.fromHisto( "score_validation",  histos, texX="F(x)", texY="a.u." )

# Plot Style
histModifications      = []
histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

ratioHistModifications = []
ratio                  = None
legend                 = (0.2,0.64,0.6,0.88)
minY                   =  1
maxY                   = (10 if model.make_log else 2.2)*max( map( lambda h:h.GetMaximum(), [training, test, training_BSM, test_BSM] ) )
yRange                 = (0,"auto") #( minY, maxY )

plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, lumi=args.luminosity, histModifications=histModifications )

##############
##############


##################
# Plot Evolution #
##################

for name, texName, test_FIs_, training_FIs_ in [
        ("all", "", test_FIs, training_FIs),
#        ("lowPt", ", p_{T}< %i GeV"%args.lowPtThresh, test_FIs_lowPt, training_FIs_lowPt),
#        ("highPt", ", p_{T} > %i GeV"%args.highPtThresh, test_FIs_highPt, training_FIs_highPt),
        ]:
    for i_tree in range(n_trees):
        test_FI_histo    .SetBinContent( i_tree+1, -sum(test_FIs_[:i_tree]) )
        training_FI_histo.SetBinContent( i_tree+1, -sum(training_FIs_[:i_tree]) )

    # Histo style
    test_FI_histo    .style = styles.lineStyle( ROOT.kBlue, width=2 )
    training_FI_histo.style = styles.lineStyle( ROOT.kRed, width=2 )
    test_FI_histo    .legendText = "Test%s"%texName
    training_FI_histo.legendText = "Training%s"%texName

    minY   = 0.01 * min( test_FI_histo.GetBinContent(test_FI_histo.GetMaximumBin()), training_FI_histo.GetBinContent(training_FI_histo.GetMaximumBin()))
    maxY   = 1.5  * max( test_FI_histo.GetBinContent(test_FI_histo.GetMaximumBin()), training_FI_histo.GetBinContent(training_FI_histo.GetMaximumBin()))

    histos = [ [test_FI_histo], [training_FI_histo] ]
    plot   = Plot.fromHisto( "FI_evolution_%s"%name,  histos, texX="b", texY="L(D,b)" )

    # Plot Style
    histModifications      = []
    histModifications      += [ lambda h: h.GetYaxis().SetTitleOffset(1.4) ]
    histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
    histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
    histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
    histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

    ratioHistModifications = []
    ratio                  = None
    if name == "all":
        legend                 = (0.6,0.75,0.9,0.88)
#        legend                 = (0.6, 0.2, 0.9, 0.32)
    elif args.model == "power_law" and name == "highPt":
        legend                 = (0.2,0.75,0.7,0.88)
    else:
        legend                 = (0.4, 0.2, 0.9, 0.4)
    yRange                 = "auto" #( minY, maxY )

    plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=False, titleOffset=0.08, histModifications=histModifications )

##############
##############


##############
# Plot Score #
##############

# Make a histogram from the score function (1D)
def score_histo( bit, title, max_n_tree = None):
    h = ROOT.TH1F(str(title), str(title), 400, model.xmin, model.xmax)
    for i in range(1, h.GetNbinsX()+1):
        h.SetBinContent( i, bit.predict([h.GetBinLowEdge(i)], max_n_tree = max_n_tree, last_tree_counts_full=False))
    return copy.deepcopy(h.Clone())

# Histo style
histos = []
histos.append( [model.score_theory] )
histos[-1][0].style      = styles.lineStyle( ROOT.kRed, width=2 )
histos[-1][0].legendText = "t(x|#theta_{0})"

# empty slot in the legend
empty = score_histo( bit, "empty", max_n_tree=0 )
empty.Scale(0)
histos.append( [empty] )
histos[-1][0].style      = styles.lineStyle( ROOT.kWhite, width=0 )
histos[-1][0].legendText = " "

counter=0
colors = [8,30,38,9,13]
for n_tree in range(bit.n_trees):
    if bit.n_trees <= n_plot or n_tree%(bit.n_trees/n_plot) == 0:

        histos.append( [score_histo( bit, str(counter), max_n_tree=n_tree )] )
        histos[-1][0].style      = styles.lineStyle( colors[counter], width=3 )
        histos[-1][0].legendText = "F^{(%i)}"%(n_tree)
        counter+=1

histos.append( [score_histo( bit, "full" )] )
histos[-1][0].style      = styles.lineStyle( ROOT.kBlack, width=3 )
histos[-1][0].legendText = "F^{(%i)}"%(n_tree+1)

# Plot of hypothesis
plot   = Plot.fromHisto( "score_boosted",  histos, texX="x", texY="F(x)" )

# Plot Style
histModifications      = []
histModifications += [ lambda h: h.GetXaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetYaxis().SetTitleSize(26) ]
histModifications += [ lambda h: h.GetXaxis().SetLabelSize(22)  ]
histModifications += [ lambda h: h.GetYaxis().SetLabelSize(22)  ]

ratioHistModifications = []
ratio                  = None
minY                   = model.min_score_theory
maxY                   = model.max_score_theory
if args.model == "power_law":
    legend                 = [(0.6,0.63,0.85,0.9),2]
elif args.model == "gaussian_width":
    legend                 = [(0.42,0.6,0.67,0.87),2]
    minY                   = -1
    maxY                   = 17
elif args.model == "gaussian_mean":
    legend                 = [(0.2,0.6,0.45,0.87),2]
elif args.model == "piece_wise":
    legend                 = [(0.65,0.15,0.9,0.5),2]
    minY                   = -0.4
    maxY                   = 0.25
elif args.model == "mixture1D":
    legend                 = [(0.2,0.45,0.45,0.73),2]
else:
    legend                 = [(0.2,0.15,0.45,0.43),2]
yRange                 = (minY, maxY)

plot1DHist( plot, plot_directory, yRange=yRange, ratio=ratio, legend=legend, lumi=args.luminosity, plotLog=False, histModifications=histModifications )

##############
##############



