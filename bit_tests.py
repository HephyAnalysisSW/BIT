#!/usr/bin/env python
# Standard imports

import numpy as np
import random
import cProfile
import time

from math import log, exp
import Analysis.Tools.syncer
import ROOT

#import uproot
#import awkward
#import pandas as pd

from BoostedInformationTree import BoostedInformationTree

from user import plot_directory

## power law
## (pT/pT0)^(theta) pT^(-alpha) -> the score is Log[pT/pT0]
#xmin  = 20
#xmax  = 420 
#theta = 0.2
#model = ROOT.TF1("model", "x^(-3)", xmin,xmax)
#score_theory = ROOT.TF1("score_theory", "1/2.-log(x/100.)", xmin, xmax)
#min_score_theory = min( score_theory.Eval(xmin), score_theory.Eval(xmax) )
#max_score_theory = max( score_theory.Eval(xmin), score_theory.Eval(xmax) ) 
#make_log  = True
#n_events      = 100000
#n_trees       = 150
#learning_rate = .5 
#max_depth     = 2
#min_size      = 50
#n_plot        = 10 
#id_   = "power-law-OT-nTrees%i-LR%3.2f-pT%iTo%i"%(n_trees, learning_rate, xmin, xmax)
#def get_dataset( n_events ):
#    features = np.array( [ [model.GetRandom(xmin, xmax)] for i in range(n_events)] )
#    weights       = np.array( [1 for i in range(n_events)] ) 
#    #diff_weights  = np.array( [ (features[i][0]/100.)**theta*log(features[i][0]/100.) for i in range(n_events)] )
#    diff_weights  = np.array( [ 1./(3-1) - log(features[i][0]/100.) for i in range(n_events)] )
#    return features, weights, diff_weights


id_       = "flat"
xmin      = 0
pT0       = 200 
xmax      = 2*pT0
model     = ROOT.TF1("model", "1", xmin, xmax)
theta     = 0.1
max_depth = 2
min_size  = 50
score_theory = ROOT.TF1("score_theory", "-1./(2+{alpha})+1./(1+{alpha})*(x>{pT0})".format(alpha=1, pT0=pT0), xmin, xmax)
min_score_theory = min( score_theory.Eval(xmin), score_theory.Eval(xmax) )
max_score_theory = max( score_theory.Eval(xmin), score_theory.Eval(xmax) ) 
make_log  = False
n_events      = 10000
n_trees       = 20
n_plot        = 5
learning_rate = 0.1
def get_dataset( n_events ):
    ''' Produces data set according to theory model'''
    features = np.array( [ [model.GetRandom(xmin, xmax)] for i in range(n_events)] )
    weights       = np.array( [ 1. for i in range(n_events)] ) 
    diff_weights  = np.array( [ -1/3.+ (1/2. if features[i]>pT0 else 0) for i in range(n_events)] )
    return features, weights, diff_weights


## Definition of the model
### exponential
#pT0=25.
#alpha=1./100
#xmin  = 20
#xmax  = 420 
#model = ROOT.TF1("model", "1./{alpha}*exp(-{alpha}*(x-{pT0}))".format(pT0=pT0, alpha=alpha), xmin, xmax)
#theta = 0.001
#score_theory = ROOT.TF1("score_theory", "1./{alpha}-(x-{pT0})".format(pT0=pT0, alpha=alpha), xmin, xmax)
#min_score_theory = min( score_theory.Eval(xmin), score_theory.Eval(xmax) )
#max_score_theory = max( score_theory.Eval(xmin), score_theory.Eval(xmax) ) 
#
#make_log  = True
#n_events      = 100000
#n_trees       = 100
#learning_rate = 0.2 
#max_depth     = 2
#min_size      = 50
#n_plot = 10 # Plot every tenth
#
#weighted = False
#id_   = "pT0%i-nTrees%i-exponential"%(pT0, n_trees)
#if weighted:id_+='-weighted'
#
#def get_sampled_dataset( n_events ):
#    features = np.array( [ [model.GetRandom(xmin, xmax)] for i in range(n_events)] )
#    weights       = np.array( [1 for i in range(n_events)] ) 
#    diff_weights  = np.array( [ (1./alpha - (features[i][0]-pT0)) for i in range(n_events)] )
#    return features, weights, diff_weights
#def get_weighted_dataset( n_events ):
#    features = np.array( [ [xmin+random.random()*(xmax-xmin)] for i in range(n_events)] )
#    weights       = np.array( [ model.Eval(features[i][0]) for i in range(n_events)] ) 
#    diff_weights  = np.array( [ weights[i]*(1./alpha - (features[i][0]-pT0)) for i in range(n_events)] )
#    return features, weights, diff_weights
#
##get_dataset = get_weighted_dataset 
#get_dataset = get_sampled_dataset 

# Produce training data set
training_features, training_weights, training_diff_weights = get_dataset( n_events )

# Let's plot the model so that Niki sees the hypothesis.
h_SM  = ROOT.TH1F("h_SM",  "h_SM",  40, 20, 420)
h_BSM = ROOT.TH1F("h_BSM", "h_BSM", 40, 20, 420)
h_BSM.SetLineColor(ROOT.kRed)
h_BSM.SetMarkerStyle(0)
for i in range(n_events):
    h_SM.Fill ( training_features[i], training_weights[i] ) 
    h_BSM.Fill( training_features[i], training_weights[i]+theta*training_diff_weights[i] ) 

# Plot of hypothesis
c1 = ROOT.TCanvas()
h_BSM.Draw("HIST")
h_SM.Draw("HISTsame")
c1.SetLogy(make_log)
c1.Print("%s/model_%s.png"%(plot_directory,id_))

bit = BoostedInformationTree(
        training_features = training_features,
        training_weights      = training_weights, 
        training_diff_weights = training_diff_weights, 
        learning_rate = learning_rate, 
        n_trees = n_trees, max_depth=max_depth, min_size=min_size, split_method='python_loop')

bit.boost()

# Make a histogram from the score function (1D)
def score_histo( bit, max_n_tree = None):
    h = ROOT.TH1F("h", "h", 400, 20, 420)
    for i in range(1, h.GetNbinsX()+1):
        h.SetBinContent( i, bit.predict([h.GetBinLowEdge(i)], max_n_tree = max_n_tree))
    return h

score_theory.SetLineColor(ROOT.kRed)
score_theory.Draw()
#score_theory.GetYaxis().SetRangeUser(.4, 2)
score_theory.GetYaxis().SetRangeUser(min_score_theory, max_score_theory)
counter=0
for n_tree in range(bit.n_trees):
    if bit.n_trees<=n_plot or n_tree%(bit.n_trees/n_plot)==0:
        fitted = score_histo( bit, max_n_tree = n_tree ) 
        fitted.SetLineColor(2+counter)
        fitted.GetYaxis().SetRangeUser(min_score_theory, max_score_theory)
        fitted.DrawCopy("HISTsame")
        counter+=1

fitted = score_histo( bit ) 
fitted.SetLineColor(ROOT.kBlack)
fitted.Draw("HISTsame")
c1.SetLogy(0)
c1.Print("%s/score_boosted_%s.png"%(plot_directory,id_))

test_features, test_weights, test_diff_weights = get_dataset( n_events )

training_profile     = ROOT.TProfile("train", "train",         20, 20, 420)#, min_score_theory, max_score_theory )
test_profile         = ROOT.TProfile("test",  "test",          20, 20, 420)#, min_score_theory, max_score_theory )
training_BSM_profile = ROOT.TProfile("BSM_train", "BSM_train", 20, 20, 420)#, min_score_theory, max_score_theory )
test_BSM_profile     = ROOT.TProfile("BSM_test",  "BSM_test",  20, 20, 420)#, min_score_theory, max_score_theory )

training     = ROOT.TH1D("train", "train",          20, min_score_theory, max_score_theory )
test         = ROOT.TH1D("test",  "test",           20, min_score_theory, max_score_theory )
training_BSM = ROOT.TH1D("BSM_train", "BSM_train",  20, min_score_theory, max_score_theory )
test_BSM     = ROOT.TH1D("BSM_test",  "BSM_test",   20, min_score_theory, max_score_theory )

training_FI_histo     = ROOT.TH1D("train", "train",          n_trees, 1, n_trees+1 )
test_FI_histo         = ROOT.TH1D("test",  "test",           n_trees, 1, n_trees+1 )

test_FIs     = np.zeros(n_trees)
training_FIs = np.zeros(n_trees)
test_FIs_lowPt     = np.zeros(n_trees)
training_FIs_lowPt = np.zeros(n_trees)
test_FIs_highPt     = np.zeros(n_trees)
training_FIs_highPt = np.zeros(n_trees)
for i in range(n_events):
    test_scores     = bit.predict( test_features[i], summed = False)
    training_scores = bit.predict( training_features[i], summed = False)

    test_score  = sum( test_scores )
    train_score = sum( training_scores )

    test_profile    .Fill(      test_features[i][0],     test_score,   test_weights[i] ) 
    training_profile.Fill(      training_features[i][0], train_score,  training_weights[i] )
    test_BSM_profile    .Fill(  test_features[i][0],     test_score,   test_weights[i]+theta*test_diff_weights[i] ) 
    training_BSM_profile.Fill(  training_features[i][0], train_score,  training_weights[i]+theta*training_diff_weights[i] )  
    test    .Fill(       test_score, test_weights[i]) 
    training.Fill(       train_score, training_weights[i])
    test_BSM    .Fill(   test_score,  test_weights[i]+theta*test_diff_weights[i]) 
    training_BSM.Fill(   train_score, training_weights[i]+theta*training_diff_weights[i]) 

    # compute test and training FI evolution during training
    test_FIs     += test_diff_weights[i]*test_scores 
    training_FIs += training_diff_weights[i]*training_scores 
    if test_features[i][0]<50:
        test_FIs_lowPt          += test_diff_weights[i]*test_scores              
    if training_features[i][0]<50:
        training_FIs_lowPt      += training_diff_weights[i]*training_scores 
    if test_features[i][0]>200:
        test_FIs_highPt         += test_diff_weights[i]*test_scores          
    if training_features[i][0]>200:
        training_FIs_highPt     += training_diff_weights[i]*training_scores 

training_profile.SetLineColor(ROOT.kRed)
test_profile    .SetLineColor(ROOT.kBlue)
training_profile.SetLineStyle(ROOT.kDashed)
test_profile    .SetLineStyle(ROOT.kDashed)
training_profile.SetMarkerStyle(0)
test_profile    .SetMarkerStyle(0)
training_BSM_profile.SetLineColor(ROOT.kRed)
test_BSM_profile    .SetLineColor(ROOT.kBlue)
training_BSM_profile.SetMarkerStyle(0)
test_BSM_profile    .SetMarkerStyle(0)

training_profile.Draw("hist")
test_profile.Draw("histsame")
training_BSM_profile.Draw("histsame")
test_BSM_profile.Draw("histsame")
c1.Print("%s/score_profile_validation_profile_%s.png"%(plot_directory,id_))

training.SetLineColor(ROOT.kBlue)
test    .SetLineColor(ROOT.kBlue)
training.SetLineStyle(ROOT.kDashed)
training.SetMarkerStyle(0)
test    .SetMarkerStyle(0)
training_BSM.SetLineColor(ROOT.kRed)
test_BSM    .SetLineColor(ROOT.kRed)
training_BSM.SetLineStyle(ROOT.kDashed)
training_BSM.SetMarkerStyle(0)
test_BSM    .SetMarkerStyle(0)

training_BSM.Draw("hist")
training_BSM.GetYaxis().SetRangeUser( (1 if make_log else 0), (3 if make_log else 1.2)*max(map( lambda h:h.GetMaximum(), [training, test, training_BSM, test_BSM]  )) )
test_BSM.Draw("histsame")
training.Draw("histsame")
test.Draw("histsame")
c1.SetLogy(make_log)
l = ROOT.TLegend(0.6, 0.74, 1.0, 0.92)
l.AddEntry(training, "train (SM)")
l.AddEntry(test, "test (SM)")
l.AddEntry(training_BSM, "train (BSM)")
l.AddEntry(test_BSM, "test (BSM)")
l.SetFillStyle(0)
l.SetShadowColor(ROOT.kWhite)
l.SetBorderSize(0)
l.Draw()
c1.Print("%s/score_validation_%s.png"%(plot_directory,id_))


for name, test_FIs_, training_FIs_ in [
        ("all", test_FIs, training_FIs),
        ("lowPt", test_FIs_lowPt, training_FIs_lowPt),
        ("highPt", test_FIs_highPt, training_FIs_highPt),
        ]:
    for i_tree in range(n_trees):
        test_FI_histo    .SetBinContent( i_tree+1, sum(test_FIs_[:i_tree]) )        
        training_FI_histo.SetBinContent( i_tree+1, sum(training_FIs_[:i_tree]) )        

    test_FI_histo    .SetLineColor(ROOT.kBlue)
    training_FI_histo.SetLineColor(ROOT.kRed)
    test_FI_histo    .Draw("hist")
    test_FI_histo    .GetYaxis().SetRangeUser(
        #1+0.5*min(test_FI_histo.GetMinimum(), training_FI_histo.GetMinimum()),
        (10**-2)*min(test_FI_histo.GetBinContent(test_FI_histo.GetMaximumBin()), training_FI_histo.GetBinContent(training_FI_histo.GetMaximumBin())),
        1.5*max(     test_FI_histo.GetBinContent(test_FI_histo.GetMaximumBin()), training_FI_histo.GetBinContent(training_FI_histo.GetMaximumBin())),
        )
    test_FI_histo    .GetXaxis().SetTitle("tree")
    training_FI_histo.Draw("histsame")
    test_FI_histo    .SetMarkerStyle(0)
    training_FI_histo.SetMarkerStyle(0)
    l = ROOT.TLegend(0.6, 0.14, 1.0, 0.23)
    l.AddEntry(training_FI_histo, "train FI (%s)"%name)
    l.AddEntry(test_FI_histo, "test FI (%s)"%name)
    l.SetFillStyle(0)
    l.SetShadowColor(ROOT.kWhite)
    l.SetBorderSize(0)
    l.Draw()
    c1.SetLogy(0)
    c1.Print("%s/FI_evolution_%s_%s.png"%(plot_directory,id_,name))

# Let's fit a single tree with different depths
#for depth in range(1,4):
#    root = Node( features, max_depth=depth, min_size=min_size, training_weights=training_weights, training_diff_weights=training_diff_weights, split_method='vectorized_split_and_weight_sums' )
#    fitted = score_histo(root)
#    fitted.SetLineColor(ROOT.kBlue)
#    fitted.Draw("HIST")
#    fitted.GetYaxis().SetRangeUser(-1.5, 1.5)
#    score_theory.SetLineColor(ROOT.kRed)
#    score_theory.Draw("same")
#    c1.SetLogy(0)
#    c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/score_depth_%i.png"%depth)

## Boost
#max_depth = 2
#min_size = 100
#score_theory.SetLineColor(ROOT.kRed)
#score_theory.Draw()
#fitted = None
#learning_rate = 0.10
#n_trees=100
#n_plot =10
#for n_tree in range(n_trees):
#
#    print "At tree", n_tree
#    # fit to data
#    root = Node( features, max_depth=max_depth, min_size=min_size, training_weights=training_weights, training_diff_weights=training_diff_weights, split_method='vectorized_split_and_weight_sums' )
#    print "max/min", max(training_diff_weights), min(training_diff_weights)
#    #root.print_tree()
#    histo = score_histo(root)
#    # Except for the last node, only take a fraction of the score
#
#    if True: #n_tree<n_trees-1:
#        histo.Scale(learning_rate)
#        # reduce the score
#        training_diff_weights+= -learning_rate*np.multiply(training_weights, np.array([root.predict(feature) for feature in features]))
#
#    if fitted is None:
#        fitted = histo 
#    else:
#        fitted.Add(histo)
#    
#    if n_tree%(n_trees/n_plot)==0: 
#        fitted.SetLineColor(ROOT.kBlue+n_tree/(n_trees/n_plot))
#        fitted.GetYaxis().SetRangeUser(-1.5, 1.5)
#        fitted.DrawCopy("HISTsame")
#
#fitted.SetLineColor(ROOT.kBlack)
#fitted.Draw("HISTsame")
#c1.SetLogy(0)
#c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/score_boosted_3.png")
#
