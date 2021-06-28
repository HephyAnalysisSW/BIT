#!/usr/bin/env python
# Standard imports
import copy
import random
import cProfile
import operator 
import time
from math import log, exp
import Analysis.Tools.syncer
import ROOT
import sys

import uproot
import awkward
import numpy as np
import pandas as pd

from it_vectorized import Node

from typing import Iterable 

def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

class BoostedInformationTree:

    def __init__( self, training_features, training_weights, training_diff_weights, n_trees = 100, learning_rate = "auto", **kwargs ):

        self.n_trees        = n_trees
        self.learning_rate  = learning_rate
       
        # Attempt to learn 98%. (1-learning_rate)^n_trees = 0.02 -> After the fit, the score is at least down to 2% 
        if learning_rate == "auto":
            self.learning_rate = 1-0.02**(1./self.n_trees)
        self.kwargs         = kwargs

        self.training_weights       = training_weights
        self.training_diff_weights  = np.copy(training_diff_weights) # Protect the outside from reweighting.
        self.training_features      = training_features

        # Will hold the trees
        self.trees                  = []

    def boost( self ):

        toolbar_width = min(20, self.n_trees)

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['


        for n_tree in range(self.n_trees):

            # fit to data
            root = Node(    self.training_features, 
                            training_weights        =   self.training_weights, 
                            training_diff_weights   =   self.training_diff_weights,
                            **self.kwargs 
                        )

            # Recall current tree
            self.trees.append( root )

            #print "max/min", max(training_diff_weights), min(training_diff_weights)
            #root.print_tree()
            #histo = score_histo(root)
            # Except for the last node, only take a fraction of the score

            #histo.Scale(learning_rate)
            # reduce the score
            self.training_diff_weights+= -self.learning_rate*np.multiply(self.training_weights, np.array([root.predict(feature) for feature in training_features]))

            # update the bar
            if self.n_trees>=toolbar_width:
                if n_tree % (self.n_trees/toolbar_width)==0:   sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("]\n") # this ends the progress bar

    #def calibrated( self ):
    #    if hasattr( self, "calibration"):
    #        minimum, maximum = self.calibration
    #    else:
    #        flatten( [ map( lambda tree: tree.get_list(only_threshold=True)
    #         
        
    def predict( self, feature_array, max_n_tree = None, summed = True):
        #return self.calibrated( sum( [ self.learning_rate*tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ], 0. ) )
        predictions = [ self.learning_rate*tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ]
        if summed: 
            return sum( predictions, 0. )
        else:
            return np.array(predictions)

    #self.training_diff_weights+= -self.learning_rate*np.multiply(self.training_weights, np.array([root.predict(feature) for feature in training_features]))

## power law
## (pT/pT0)^(theta) pT^(-alpha) -> the score is Log[pT/pT0]
model = ROOT.TF1("model", "x^(-3)", 20, 1000)
theta = 0.2
score_theory = ROOT.TF1("score_theory", "log(x/100.)", 20, 420)
max_score_theory = score_theory.Eval(420)
min_score_theory = score_theory.Eval(20)
make_log  = True
n_events      = 100000
n_trees       = 100
learning_rate = 0.2 
max_depth     = 3
min_size      = 50
n_plot        = 10 # Plot every tenth
id_   = "power-law-OT-nTrees%i"%n_trees
def get_dataset( n_events ):
    features = np.array( [ [model.GetRandom(20, 420)] for i in range(n_events)] )
    weights       = np.array( [1 for i in range(n_events)] ) 
    diff_weights  = np.array( [ (features[i][0]/100.)**theta*log(features[i][0]/100.) for i in range(n_events)] )
    return features, weights, diff_weights

# Issues: 1. Occasional problems with accidental dtype='int' -> should be float
#         2. Deal with the pathological case where all delta-FI=0 when splitting. (We don't always want to split) 

# (pT/pT0)^(theta) pT^(-alpha) -> the score is Log[pT/pT0]
#id_       = "flat"
#model     = ROOT.TF1("model", "1", 20, 1000)
#theta     = 0.1
#max_depth = 1
#pT0       = 220 # model parameter
#score_theory = ROOT.TF1("score_theory", "(x>={pT0})".format(pT0=pT0), 20, 420)
#max_score_theory = score_theory.Eval(420)
#min_score_theory = score_theory.Eval(20)
#make_log  = False
#n_events      = 100000
#n_trees       = 1
#learning_rate = 0.5
#def get_dataset( n_events ):
#    ''' Produces data set according to theory model'''
#    features = np.array( [ [model.GetRandom(20, 420)] for i in range(n_events)] )
#    weights       = np.array( [ 1. for i in range(n_events)] ) 
#    diff_weights  = np.array( [ 0. if features[i]<pT0 else 1 for i in range(n_events)] )
#    return features, weights, diff_weights

        

# Definition of the model
## exponential
## Exp[-pT/pT0] -> the score is pT/(pT0*(1+theta)**2)
#pT0=25.
#model = ROOT.TF1("model", "exp(-x/{pT0})*exp({pT0}/20.)".format(pT0=pT0), 20, 1000)
#theta = 0.1
#score_theory = ROOT.TF1("score_theory", "x/({pT0}*(1+{theta})**2)".format(pT0=pT0,theta=theta), 20, 420)
#max_score_theory = score_theory.Eval(420)
#min_score_theory = score_theory.Eval(20)

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
#    features = np.array( [ [model.GetRandom(20, 420)] for i in range(n_events)] )
#    weights       = np.array( [1 for i in range(n_events)] ) 
#    diff_weights  = np.array( [ features[i][0]/(pT0*(1+theta)**2)*exp(features[i][0]*theta/(pT0*(1+theta))) for i in range(n_events)] )
#    return features, weights, diff_weights
#def get_weighted_dataset( n_events ):
#    features = np.array( [ [20+random.random()*400] for i in range(n_events)] )
#    weights       = np.array( [ model.Eval(features[i][0]) for i in range(n_events)] ) 
#    diff_weights  = np.array( [ weights[i]*features[i][0]/(pT0*(1+theta)**2)*exp(features[i][0]*theta/(pT0*(1+theta))) for i in range(n_events)] )
#    return features, weights, diff_weights
#
#get_dataset = get_weighted_dataset 

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
c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/model_%s.png"%id_)

bit = BoostedInformationTree(
        training_features = training_features,
        training_weights      = training_weights, 
        training_diff_weights = training_diff_weights, 
        learning_rate = learning_rate, 
        n_trees = n_trees, max_depth=max_depth, min_size=min_size )

bit.boost()

# Make a histogram from the score function (1D)
def score_histo( bit, max_n_tree = None):
    h = ROOT.TH1F("h", "h", 400, 20, 420)
    for i in range(1, h.GetNbinsX()+1):
        h.SetBinContent( i, bit.predict([h.GetBinLowEdge(i)], max_n_tree = max_n_tree))
    return h

score_theory.SetLineColor(ROOT.kRed)
score_theory.Draw()
counter=0
for n_tree in range(bit.n_trees):
    if bit.n_trees<=n_plot or n_tree%(bit.n_trees/n_plot)==0:
        fitted = score_histo( bit, max_n_tree = n_tree ) 
        fitted.SetLineColor(ROOT.kBlue+counter)
        fitted.GetYaxis().SetRangeUser(min_score_theory, max_score_theory)
        fitted.DrawCopy("HISTsame")
        counter+=1

fitted = score_histo( bit ) 
fitted.SetLineColor(ROOT.kBlack)
fitted.Draw("HISTsame")
c1.SetLogy(0)
c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/score_boosted_%s.png"%id_)

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
c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/score_profile_validation_profile_%s.png"%id_)

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
c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/score_validation_%s.png"%id_)


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
    c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/FI_evolution_%s_%s.png"%(id_,name))

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
