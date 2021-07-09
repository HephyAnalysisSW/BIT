#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
from math import log, exp

# load root macro
ROOT.gROOT.LoadMacro('$CMSSW_BASE/src/Analysis/Tools/scripts/tdrstyle.C')
ROOT.setTDRStyle()
ROOT.gROOT.LoadMacro("$CMSSW_BASE/src/BIT/tdrstyles/anotherNiceColorPalette.C")
#ROOT.anotherNiceColorPalette(20)
ROOT.niceColorPalette(15)

# Analysis
import Analysis.Tools.syncer

# RootTools
from RootTools.core.standard   import *
from RootTools.plot.helpers    import copyIndexPHP

# BIT
from BoostedInformationTree import BoostedInformationTree

# Plot helpers
from plot_helpers import plot2DHist

# User
from user import plot_directory as user_plot_directory


# Model choices
allModels = set( [ os.path.splitext(item)[0] for item in os.listdir( "toy_models" ) if not item.startswith("_") and "2D" in item ] )

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="BIT_v1",                                                                help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="exponential_2D", type=str,   choices=allModels,                         help="import model")
argParser.add_argument("--nTraining",          action="store",      default=100000,           type=int,                                              help="number of training events")
argParser.add_argument("--luminosity",         action="store",      default=137,              type=int,                                              help="luminosity value, currently only for plot label")
argParser.add_argument("--treeRange",          action="store",      default=[1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], type=int, nargs='*', help="list of nTrees for plots")
args = argParser.parse_args()

# import the toy model
import toy_models as models
model = getattr( models, args.model )

# Produce training data set
training_features, training_weights, training_diff_weights = model.get_dataset( args.nTraining )

# directory for plots
plot_directory = os.path.join( user_plot_directory, args.plot_directory, model.id_string )

if not os.path.isdir(plot_directory):
    os.makedirs( plot_directory )

time1 = time.time()

# BIT config
n_trees       = model.n_trees
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
        split_method          = 'vectorized_split_and_weight_sums',
        weights_update_method = 'vectorized')

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

c1 = ROOT.TCanvas("can","",550,600)

pads = ROOT.TPad("pad","",0.,0.,1.,1.)
pads.SetRightMargin(0.02)
pads.SetLeftMargin(0.14)
pads.SetTopMargin(0.08)
pads.Draw()
pads.cd()

model.score_theory.GetXaxis().SetRangeUser(model.xmin, model.xmax)
model.score_theory.GetYaxis().SetRangeUser(model.ymin, model.ymax)
model.score_theory.GetXaxis().SetTitle(model.texX)
model.score_theory.GetYaxis().SetTitle(model.texX)

model.score_theory.GetXaxis().SetTitleFont(42)
model.score_theory.GetYaxis().SetTitleFont(42)
model.score_theory.GetXaxis().SetLabelFont(42)
model.score_theory.GetYaxis().SetLabelFont(42)

model.score_theory.GetXaxis().SetTitleOffset(1.1)
model.score_theory.GetYaxis().SetTitleOffset(1.45)

model.score_theory.GetXaxis().SetTitleSize(0.045)
model.score_theory.GetYaxis().SetTitleSize(0.045)
model.score_theory.GetXaxis().SetLabelSize(0.04)
model.score_theory.GetYaxis().SetLabelSize(0.04)

model.score_theory.SetLineWidth(2)
model.score_theory.SetLineStyle(7)
model.score_theory.SetLineColor(ROOT.kGray+2)

counter=0
for n_tree in args.treeRange:

    print "n_tree", n_tree
    fitted = score_histo( bit, max_n_tree = n_tree )
    fitted.SetLineWidth(3)
    fitted.SetLineColor(ROOT.kBlue)

    fitted.Smooth()

    model.score_theory.Draw("CONT3")
    fitted.DrawCopy("CONT1SAME")

    c1.SetLogy(0)

    leg = ROOT.TLegend(0.15,0.72,0.81,0.87)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.037)
    leg.SetFillStyle(0)
    leg.AddEntry( model.score_theory, "Score (theory)","l" )
    leg.AddEntry( fitted,             "Score (boosted, N_{tree}= %i)"%(n_tree), "l" )
#    leg.Draw()


    latex1 = ROOT.TLatex()
    latex1.SetNDC()
    latex1.SetTextSize(0.04)
    latex1.SetTextFont(42)
    latex1.SetTextAlign(11)

    latex2 = ROOT.TLatex()
    latex2.SetNDC()
    latex2.SetTextSize(0.05)
    latex2.SetTextFont(42)
    latex2.SetTextAlign(11)

    latex2.DrawLatex(0.14, 0.94, '#bf{Boosted Info Trees}'),
    latex1.DrawLatex(0.7, 0.94, '#bf{%i fb^{-1} (13 TeV)}' %args.luminosity)

    for e in [".png",".pdf",".root"]:
        c1.Print(os.path.join(plot_directory, "score_boosted_nTreePlotted%i%s"%(n_tree,e)))


copyIndexPHP(plot_directory)
