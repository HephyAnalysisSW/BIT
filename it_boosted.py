#!/usr/bin/env python
# Standard imports
import copy
import cProfile
import operator 
import time
from math import log
import Analysis.Tools.syncer
import ROOT
import sys

import uproot
import awkward
import numpy as np
import pandas as pd

from it_vectorized import Node

class BoostedInformationTree:

    def __init__( self, features, training_weights, training_diff_weights, learning_rate = 0.1, n_trees = 100, max_depth=2, min_size=50 ):

        self.learning_rate  = learning_rate
        self.n_trees        = n_trees
        self.max_depth      = max_depth
        self.min_size       = min_size


        self.training_weights       = training_weights
        self.training_diff_weights  = np.copy(training_diff_weights) # Protect the outside from reweighting.
        self.features               = features

        # Will hold the trees
        self.trees                  = []

    def boost( self ):

        toolbar_width = 20

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['


        for n_tree in range(self.n_trees):

            # fit to data
            root = Node(    self.features, 
                            max_depth   =   self.max_depth, 
                            min_size    =   self.min_size, 
                            training_weights        =   self.training_weights, 
                            training_diff_weights   =   self.training_diff_weights, 
                            split_method=   'vectorized_split_and_weight_sums' )

            # Recall current tree
            self.trees.append( root )

            #print "max/min", max(training_diff_weights), min(training_diff_weights)
            #root.print_tree()
            #histo = score_histo(root)
            # Except for the last node, only take a fraction of the score

            #histo.Scale(learning_rate)
            # reduce the score
            self.training_diff_weights+= -self.learning_rate*np.multiply(self.training_weights, np.array([root.predict(feature) for feature in features]))

            # update the bar
            if n_tree % (self.n_trees/toolbar_width)==0:   sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("]\n") # this ends the progress bar

    def predict( self, feature_array, max_n_tree = None):
        return sum( [ self.learning_rate*tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ], 0. )
        

# model
# (pT/pT0)^(theta) pT^(-alpha) -> the score is Log[pT/pT0]

max_events  = 100000
model    = ROOT.TF1("model", "x^(-3)", 20, 1000)
features = np.array( [ [model.GetRandom(20, 420)] for i in range(max_events)] )
training_weights       = np.array( [1 for i in range(max_events)] ) 
theta    = 0.1
training_diff_weights  = np.array( [ (features[i][0]/100.)**theta*log(features[i][0]/100.) for i in range(max_events)] )

# for convinience compute the score
score_theory = ROOT.TF1("score_theory", "log(x/100.)", 20, 420)

# Let's plot the model so that Niki sees the hypothesis.
h_SM  = ROOT.TH1F("h_SM", "h_SM", 40, 20, 420)
h_BSM = ROOT.TH1F("h_BSM", "h_BSM", 40, 20, 420)
h_BSM.SetLineColor(ROOT.kRed)
h_BSM.SetMarkerStyle(0)
for i in range(max_events):
    h_SM.Fill ( features[i], training_weights[i] ) 
    h_BSM.Fill( features[i], training_weights[i]+theta*training_diff_weights[i] ) 

c1 = ROOT.TCanvas()
h_SM.Draw("HIST")
h_BSM.Draw("HISTsame")
c1.SetLogy()
c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/model_3.png")

bit = BoostedInformationTree(
        features = features,
        training_weights      = training_weights, 
        training_diff_weights = training_diff_weights, 
        learning_rate = 0.1, 
        n_trees = 100, max_depth=2, min_size=50 )

bit.boost()

# Make a histogram from the score function (1D)
def score_histo( bit, max_n_tree = None):
    h = ROOT.TH1F("h", "h", 400, 20, 420)
    for i in range(1, h.GetNbinsX()+1):
        h.SetBinContent( i, bit.predict([h.GetBinLowEdge(i)], max_n_tree = max_n_tree))
    return h

score_theory.SetLineColor(ROOT.kRed)
score_theory.Draw()
n_plot = 10 # Plot every tenth
for n_tree in range(bit.n_trees):
    if n_tree%(bit.n_trees/n_plot)==0:
        fitted = score_histo( bit, max_n_tree = n_tree ) 
        fitted.SetLineColor(ROOT.kBlue+n_tree/(bit.n_trees/n_plot))
        fitted.GetYaxis().SetRangeUser(-1.5, 1.5)
        fitted.DrawCopy("HISTsame")

fitted = score_histo( bit ) 
fitted.SetLineColor(ROOT.kBlack)
fitted.Draw("HISTsame")
c1.SetLogy(0)
c1.Print("/mnt/hephy/cms/robert.schoefbeck/www/etc/score_boosted_3.png")


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
