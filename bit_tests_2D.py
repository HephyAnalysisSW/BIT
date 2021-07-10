#!/usr/bin/env python
# Standard imports

import numpy as np
import random
import cProfile
import time
import os

from math import log, exp
import Analysis.Tools.syncer
import ROOT

from BoostedInformationTree import BoostedInformationTree

import time

# import the toy model
from toy_models import exponential_2D as model
theta    = 0.001

#from toy_models import power_law as model
#theta    = 0.2

# Produce training data set
n_events = 100000
training_features, training_weights, training_diff_weights = model.get_dataset( n_events )

# directory for plots
from user import plot_directory as user_plot_directory
plot_directory = os.path.join( user_plot_directory, model.id_string )

if not os.path.isdir(plot_directory):
    os.makedirs( plot_directory )

## Let's plot the model so that Niki sees the hypothesis.
#h_SM  = ROOT.TH1F("h_SM",  "h_SM",  40, model.xmin, model.xmax)
#h_BSM = ROOT.TH1F("h_BSM", "h_BSM", 40, model.xmin, model.xmax)
#h_BSM.SetLineColor(ROOT.kRed)
#h_BSM.SetMarkerStyle(0)
#for i in range(n_events):
#    h_SM.Fill ( training_features[i], training_weights[i] ) 
#    h_BSM.Fill( training_features[i], training_weights[i]+theta*training_diff_weights[i] ) 
#
## Plot of hypothesis
#h_BSM.Draw("HIST")
#h_SM.Draw("HISTsame")
#c1.SetLogy(model.make_log)

#c1.Print(os.path.join(plot_directory, "model.png"))

time1 = time.time()

# BIT config
n_trees       = model.n_trees
max_depth     = 2 
learning_rate = 0.20
min_size      = 100
n_plot        = 5

bit= BoostedInformationTree(
        training_features = training_features,
        training_weights      = training_weights, 
        training_diff_weights = training_diff_weights, 
        learning_rate = learning_rate, 
        n_trees = n_trees,
        max_depth=max_depth,
        min_size=min_size,
        split_method='vectorized_split_and_weight_sums',
        weights_update_method='vectorized')

bit.boost()
time2 = time.time()
boosting_time = time2 - time1

print "Boosting time: %.2f seconds" % boosting_time

# Make a histogram from the score function (1D)
def score_histo( bit, max_n_tree = None):
    h = ROOT.TH2F("h", "h", 400, 20, 420, 400, 20, 420)
    for i in range(1, h.GetNbinsX()+1):
        for j in range(1, h.GetNbinsY()+1):
            h.SetBinContent( h.FindBin(h.GetXaxis().GetBinLowEdge(i), h.GetYaxis().GetBinLowEdge(j)), bit.predict([i, j], max_n_tree = max_n_tree))
    return h

#n_contour = 10
#model.score_theory.SetContour(10)
#for i in range(10):
#    model.score_theory.SetContourLevel(i)

c1 = ROOT.TCanvas()
#model.score_theory.SetLineColor(ROOT.kRed)
#score_theory.GetYaxis().SetRangeUser(.4, 2)
model.score_theory.GetXaxis().SetRangeUser(model.xmin, model.xmax)
model.score_theory.GetYaxis().SetRangeUser(model.ymin, model.ymax)
counter=0
for n_tree in [1, 2, 5, 10, 20, 30, 40, 50, 80, 100, 200, 1000]:

    print "n_tree", n_tree
    fitted = score_histo( bit, max_n_tree = n_tree ) 
#        fitted.SetLineColor(2+counter)
#        #fitted.GetYaxis().SetRangeUser(model.min_score_theory, model.max_score_theory)
#        counter+=1

    fitted.DrawCopy("CONT3")
    model.score_theory.Draw("CONT3SAME")

    c1.SetLogy(0)
    c1.Print(os.path.join(plot_directory, "score_boosted_nTreePlotted%i.png"%n_tree))

#test_features, test_weights, test_diff_weights = model.get_dataset( n_events )
#
#training_profile     = ROOT.TProfile("train", "train",         20, model.xmin, model.xmax) 
#test_profile         = ROOT.TProfile("test",  "test",          20, model.xmin, model.xmax)
#training_BSM_profile = ROOT.TProfile("BSM_train", "BSM_train", 20, model.xmin, model.xmax)
#test_BSM_profile     = ROOT.TProfile("BSM_test",  "BSM_test",  20, model.xmin, model.xmax) 
#
#training     = ROOT.TH1D("train", "train",          20, model.min_score_theory, model.max_score_theory )
#test         = ROOT.TH1D("test",  "test",           20, model.min_score_theory, model.max_score_theory )
#training_BSM = ROOT.TH1D("BSM_train", "BSM_train",  20, model.min_score_theory, model.max_score_theory )
#test_BSM     = ROOT.TH1D("BSM_test",  "BSM_test",   20, model.min_score_theory, model.max_score_theory )
#
#training_FI_histo     = ROOT.TH1D("train", "train",          n_trees, 1, n_trees+1 )
#test_FI_histo         = ROOT.TH1D("test",  "test",           n_trees, 1, n_trees+1 )
#
#test_FIs     = np.zeros(n_trees)
#training_FIs = np.zeros(n_trees)
#test_FIs_lowPt     = np.zeros(n_trees)
#training_FIs_lowPt = np.zeros(n_trees)
#test_FIs_highPt     = np.zeros(n_trees)
#training_FIs_highPt = np.zeros(n_trees)
#for i in range(n_events):
#    test_scores     = bit.predict( test_features[i], summed = False)
#    training_scores = bit.predict( training_features[i], summed = False)
#
#    test_score  = sum( test_scores )
#    train_score = sum( training_scores )
#
#    test_profile    .Fill(      test_features[i][0],     test_score,   test_weights[i] ) 
#    training_profile.Fill(      training_features[i][0], train_score,  training_weights[i] )
#    test_BSM_profile    .Fill(  test_features[i][0],     test_score,   test_weights[i]+theta*test_diff_weights[i] ) 
#    training_BSM_profile.Fill(  training_features[i][0], train_score,  training_weights[i]+theta*training_diff_weights[i] )  
#    test    .Fill(       test_score, test_weights[i]) 
#    training.Fill(       train_score, training_weights[i])
#    test_BSM    .Fill(   test_score,  test_weights[i]+theta*test_diff_weights[i]) 
#    training_BSM.Fill(   train_score, training_weights[i]+theta*training_diff_weights[i]) 
#
#    # compute test and training FI evolution during training
#    test_FIs     += test_diff_weights[i]*test_scores 
#    training_FIs += training_diff_weights[i]*training_scores 
#    if test_features[i][0]<50:
#        test_FIs_lowPt          += test_diff_weights[i]*test_scores              
#    if training_features[i][0]<50:
#        training_FIs_lowPt      += training_diff_weights[i]*training_scores 
#    if test_features[i][0]>200:
#        test_FIs_highPt         += test_diff_weights[i]*test_scores          
#    if training_features[i][0]>200:
#        training_FIs_highPt     += training_diff_weights[i]*training_scores 
#
#training_profile.SetLineColor(ROOT.kRed)
#test_profile    .SetLineColor(ROOT.kBlue)
#training_profile.SetLineStyle(ROOT.kDashed)
#test_profile    .SetLineStyle(ROOT.kDashed)
#training_profile.SetMarkerStyle(0)
#test_profile    .SetMarkerStyle(0)
#training_BSM_profile.SetLineColor(ROOT.kRed)
#test_BSM_profile    .SetLineColor(ROOT.kBlue)
#training_BSM_profile.SetMarkerStyle(0)
#test_BSM_profile    .SetMarkerStyle(0)
#
#training_profile.Draw("hist")
#test_profile.Draw("histsame")
#training_BSM_profile.Draw("histsame")
#test_BSM_profile.Draw("histsame")
#c1.Print(os.path.join(plot_directory, "score_profile_validation_profile.png"))
#
#training.SetLineColor(ROOT.kBlue)
#test    .SetLineColor(ROOT.kBlue)
#training.SetLineStyle(ROOT.kDashed)
#training.SetMarkerStyle(0)
#test    .SetMarkerStyle(0)
#training_BSM.SetLineColor(ROOT.kRed)
#test_BSM    .SetLineColor(ROOT.kRed)
#training_BSM.SetLineStyle(ROOT.kDashed)
#training_BSM.SetMarkerStyle(0)
#test_BSM    .SetMarkerStyle(0)
#
#training_BSM.Draw("hist")
#training_BSM.GetYaxis().SetRangeUser( (1 if model.make_log else 0), (3 if model.make_log else 1.2)*max(map( lambda h:h.GetMaximum(), [training, test, training_BSM, test_BSM]  )) )
#test_BSM.Draw("histsame")
#training.Draw("histsame")
#test.Draw("histsame")
#c1.SetLogy(model.make_log)
#l = ROOT.TLegend(0.6, 0.74, 1.0, 0.92)
#l.AddEntry(training, "train (SM)")
#l.AddEntry(test, "test (SM)")
#l.AddEntry(training_BSM, "train (BSM)")
#l.AddEntry(test_BSM, "test (BSM)")
#l.SetFillStyle(0)
#l.SetShadowColor(ROOT.kWhite)
#l.SetBorderSize(0)
#l.Draw()
#c1.Print(os.path.join(plot_directory, "score_validation.png"))
#
#
#for name, test_FIs_, training_FIs_ in [
#        ("all", test_FIs, training_FIs),
#        ("lowPt", test_FIs_lowPt, training_FIs_lowPt),
#        ("highPt", test_FIs_highPt, training_FIs_highPt),
#        ]:
#    for i_tree in range(n_trees):
#        test_FI_histo    .SetBinContent( i_tree+1, sum(test_FIs_[:i_tree]) )        
#        training_FI_histo.SetBinContent( i_tree+1, sum(training_FIs_[:i_tree]) )        
#
#    test_FI_histo    .SetLineColor(ROOT.kBlue)
#    training_FI_histo.SetLineColor(ROOT.kRed)
#    test_FI_histo    .Draw("hist")
#    test_FI_histo    .GetYaxis().SetRangeUser(
#        #1+0.5*min(test_FI_histo.GetMinimum(), training_FI_histo.GetMinimum()),
#        (10**-2)*min(test_FI_histo.GetBinContent(test_FI_histo.GetMaximumBin()), training_FI_histo.GetBinContent(training_FI_histo.GetMaximumBin())),
#        1.5*max(     test_FI_histo.GetBinContent(test_FI_histo.GetMaximumBin()), training_FI_histo.GetBinContent(training_FI_histo.GetMaximumBin())),
#        )
#    test_FI_histo    .GetXaxis().SetTitle("tree")
#    training_FI_histo.Draw("histsame")
#    test_FI_histo    .SetMarkerStyle(0)
#    training_FI_histo.SetMarkerStyle(0)
#    l = ROOT.TLegend(0.6, 0.14, 1.0, 0.23)
#    l.AddEntry(training_FI_histo, "train FI (%s)"%name)
#    l.AddEntry(test_FI_histo, "test FI (%s)"%name)
#    l.SetFillStyle(0)
#    l.SetShadowColor(ROOT.kWhite)
#    l.SetBorderSize(0)
#    l.Draw()
#    c1.SetLogy(0)
#    c1.Print(os.path.join(plot_directory, "FI_evolution_%s.png"%name))
#
