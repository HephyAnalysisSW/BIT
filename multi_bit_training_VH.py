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
import pickle
import itertools

# RootTools
from RootTools.core.standard   import *
ROOT.gROOT.LoadMacro("$CMSSW_BASE/src/Analysis/Tools/scripts/tdrstyle.C")
ROOT.setTDRStyle()

import helpers

# Analysis
import Analysis.Tools.syncer as syncer

# BIT
from MultiBoostedInformationTree import MultiBoostedInformationTree

# User
import user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="multiBIT_VH",                 help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="ZH_Nakamura_debug",                 help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default=None, type=str,  help="prefix")
argParser.add_argument("--nTraining",          action="store",      default=500000,        type=int,  help="number of training events")
argParser.add_argument("--coefficients",       action="store",      default=['cHW'],       nargs="*", help="Which coefficients?")
argParser.add_argument('--overwrite',          action='store',      default=None, choices = [None, "training", "data", "all"],  help="Overwrite output?")
argParser.add_argument('--debug',              action='store_true', help="Make debug plots?")
argParser.add_argument('--feature_plots',      action='store_true', help="Feature plots?")

args, extra = argParser.parse_known_args(sys.argv[1:])

def parse_value( s ):
    try:
        r = int( s )
    except ValueError:
        try:
            r = float(s)
        except ValueError:
            r = s
    return r

extra_args = {}
key        = None
for arg in extra:
    if arg.startswith('--'):
        # previous no value? -> Interpret as flag
        #if key is not None and extra_args[key] is None:
        #    extra_args[key]=True
        key = arg.lstrip('-')
        extra_args[key] = True # without values, interpret as flag
        continue
    else:
        if type(extra_args[key])==type([]):
            extra_args[key].append( parse_value(arg) )
        else:
            extra_args[key] = [parse_value(arg)]
for key, val in extra_args.iteritems():
    if type(val)==type([]) and len(val)==1:
        extra_args[key]=val[0]

# import the VH model
import VH_models
model = getattr(VH_models, args.model)
model.multi_bit_cfg.update( extra_args )

feature_names = model.feature_names

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model )

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

# initiate plot
Plot.setDefaults()

training_data_filename = os.path.join(user.data_directory, args.model, "training_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(training_data_filename):
    training_features = model.getEvents(args.nTraining)
    training_weights  = model.getWeights(training_features, eft=model.default_eft_parameters)
    print ("Created data set of size %i" % len(training_features) )
    if not os.path.exists(os.path.dirname(training_data_filename)):
        os.makedirs(os.path.dirname(training_data_filename))
    pickle.dump( [training_features, training_weights], file(training_data_filename, 'w'))
    print "Written training data to", training_data_filename
else:
    print "Loading training data from", training_data_filename
    training_features, training_weights = pickle.load( file(training_data_filename))


#if args.coefficients is not None:
#    for key in training_weights.keys():
#        if not all( [k in args.coefficients for k in key]):
#            del training_weights[key]
#
print "nEvents: %i Weights: %s" %( len(training_features), [ k for k in training_weights.keys() if k!=tuple()] )

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
def plot1DHist( plot, plot_directory, yRange=(0.3,"auto"), ratio={'yRange':(0.1,1.9)}, legend=(0.2,0.63,0.9,0.95), lumi=137, plotLog=True, histModifications=[], titleOffset=0 ):

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
                       legend = legend,
                       histModifications = histModifications,
                       copyIndexPHP = True,
                       )

###############
## Plot Model #
###############

if args.feature_plots and hasattr( model, "eft_plot_points"):
    h    = {}
    h_lin= {}
    h_rw = {}
    h_rw_lin = {}
    for i_eft, eft_plot_point in enumerate(model.eft_plot_points):
        eft = eft_plot_point['eft']

        if i_eft == 0:
            eft_sm     = eft
        name = ''
        name= '_'.join( [ (wc+'_%3.2f'%eft[wc]).replace('.','p').replace('-','m') for wc in model.wilson_coefficients if eft.has_key(wc) ])
        tex_name = eft_plot_point['tex'] 

        if i_eft==0: name='SM'
        h[name] = {}
        eft['name']=name
        
        for i_feature, feature in enumerate(feature_names):
            h[name][feature]        = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *model.plot_options[feature]['binning'] )

        # make reweights for x-check
        reweight = copy.deepcopy(training_weights[()])
        # linear term
        for param1 in model.wilson_coefficients:
            reweight += (eft[param1]-eft_sm[param1])*training_weights[(param1,)] 
        # quadratic term
        for param1 in model.wilson_coefficients:
            if eft[param1]-eft_sm[param1] ==0: continue
            for param2 in model.wilson_coefficients:
                if eft[param2]-eft_sm[param2] ==0: continue
                reweight += .5*(eft[param1]-eft_sm[param1])*(eft[param2]-eft_sm[param2])*training_weights[tuple(sorted((param1,param2)))]

        sign_postfix = ""
        if False:
            reweight_sign = np.sign(np.sin(2*np.arccos(training_features[:,feature_names.index('cos_theta')]))*np.sin(2*np.arccos(training_features[:,feature_names.index('cos_theta_hat')])))
            reweight     *= reweight_sign
            #reweight_lin_sign = reweight_sign*reweight_lin
            sign_postfix    = " weighted with sgn(sin(2#theta)sin(2#hat{#theta}))"

        for i_feature, feature in enumerate(feature_names):
            binning = model.plot_options[feature]['binning']

            h[name][feature] = helpers.make_TH1F( np.histogram(training_features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight) )
            h[name][feature].style      = styles.lineStyle( eft_plot_point['color'], width=2, dashed=False )
            h[name][feature].legendText = tex_name

    for i_feature, feature in enumerate(feature_names):

        for _h in [h]:
            norm = _h[model.eft_plot_points[0]['eft']['name']][feature].Integral()
            if norm>0:
                for eft_plot_point in model.eft_plot_points:
                    _h[eft_plot_point['eft']['name']][feature].Scale(1./norm) 

        histos = [[h[eft_plot_point['eft']['name']][feature]] for eft_plot_point in reversed(model.eft_plot_points)]
        plot   = Plot.fromHisto( feature+'_nom',  histos, texX=model.plot_options[feature]['tex'], texY="1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'] )

        for log in [True, False]:

            # Add subdirectory for lin/log plots
            plot_directory_ = os.path.join( plot_directory, "feature_plots", "nTraining_%i"%args.nTraining, "log" if log else "lin" )
            for p in [plot] :#, plot_lin, plot_rw_lin]:
                plotting.draw( p,
                               plot_directory = plot_directory_,
                               logX = False, logY = log, sorting = False,
                               yRange = "auto" if not log else (0.002,"auto"),
                               ratio = None,
        #                       drawObjects = drawObjects( lumi, offset=titleOffset ),
                                legend=[(0.2,0.68,0.9,0.91),2],
                               #histModifications = histModifications,
                               copyIndexPHP = True,
                               )
print ("Done with plots")
syncer.sync()

base_points = []
for comb in list(itertools.combinations_with_replacement(args.coefficients,1))+list(itertools.combinations_with_replacement(args.coefficients,2)):
    base_points.append( {c:comb.count(c) for c in args.coefficients} )

if args.prefix == None:
    bit_name = "multiBit_%s_%s_nTraining_%i_nTrees_%i"%(args.model, "_".join(args.coefficients), args.nTraining, model.multi_bit_cfg["n_trees"])
else:
    bit_name = "multiBit_%s_%s_%s_nTraining_%i_nTrees_%i"%(args.model, args.prefix, "_".join(args.coefficients), args.nTraining, model.multi_bit_cfg["n_trees"])

filename = os.path.join(user.model_directory, bit_name)+'.pkl'
try:
    print ("Loading %s for %s"%(bit_name, filename))
    bit = MultiBoostedInformationTree.load(filename)
except IOError:
    bit = None

if bit is None or args.overwrite in ["all", "training"]:
    time1 = time.time()
    bit = MultiBoostedInformationTree(
            training_features     = training_features,
            training_weights      = training_weights,
            base_points           = base_points,
            feature_names         = model.feature_names,
            **model.multi_bit_cfg
                )

    bit.boost()
    bit.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    boosting_time = time2 - time1
    print ("Boosting time: %.2f seconds" % boosting_time)

test_data_filename = os.path.join(user.data_directory, args.model, "test_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(test_data_filename):
    test_features = model.getEvents(args.nTraining)
    test_weights  = model.getWeights(test_features, eft=model.default_eft_parameters)
    print ("Created data set of size %i" % len(test_features) )
    if not os.path.exists(os.path.dirname(test_data_filename)):
        os.makedirs(os.path.dirname(test_data_filename))
    pickle.dump( [test_features, test_weights], file(test_data_filename, 'w'))
    print "Written test data to", test_data_filename
else:
    print "Loading test data from", test_data_filename
    test_features, test_weights = pickle.load( file(test_data_filename))

    if args.debug:

        # Loss plot
        training_losses = helpers.make_TH1F((bit.losses(training_features, training_weights),None), ignore_binning = True)
        test_losses     = helpers.make_TH1F((bit.losses(test_features, test_weights),None),         ignore_binning = True)

        c1 = ROOT.TCanvas("c1");

        l = ROOT.TLegend(0.2,0.8,0.9,0.85)
        l.SetNColumns(2)
        l.SetFillStyle(0)
        l.SetShadowColor(ROOT.kWhite)
        l.SetBorderSize(0)


        training_losses.GetXaxis().SetTitle("N_{B}")
        training_losses.GetYaxis().SetTitle("Loss")
        l.AddEntry(training_losses, "train")
        l.AddEntry(test_losses, "test")


        test_losses.SetLineWidth(2)
        test_losses.SetLineColor(ROOT.kRed+2)
        test_losses.SetMarkerColor(ROOT.kRed+2)
        test_losses.SetMarkerStyle(0)
        training_losses.SetLineWidth(2)
        training_losses.SetLineColor(ROOT.kRed+2)
        training_losses.SetMarkerColor(ROOT.kRed+2)
        training_losses.SetMarkerStyle(0)

        training_losses.Draw("hist") 
        test_losses.Draw("histsame")

        for logY in [True, False]:
            plot_directory_ = os.path.join( plot_directory, "training_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
            c1.Print(os.path.join(plot_directory_, "loss.png"))

        # GIF animation
        tex = ROOT.TLatex()
        tex.SetNDC()
        tex.SetTextSize(0.06)

        for max_n_tree in range(1,10)+range(10,bit.n_trees+1,10):
            if max_n_tree==0: max_n_tree=1
            stuff = []
            test_predictions = bit.vectorized_predict(test_features, max_n_tree = max_n_tree)

            # colors
            color = {}
            i_lin, i_diag, i_mixed = 0,0,0
            for i_der, der in enumerate(bit.derivatives):
                if len(der)==1:
                    color[der] = ROOT.kAzure + i_lin
                    i_lin+=1
                elif len(der)==2 and len(set(der))==1:
                    color[der] = ROOT.kRed + i_diag
                    i_diag+=1
                elif len(der)==2 and len(set(der))==2:
                    color[der] = ROOT.kGreen + i_mixed
                    i_mixed+=1

            w0 = test_weights[()]
            h_w0, h_ratio_prediction, h_ratio_truth, lin_binning = {}, {}, {}, {}
            for i_feature, feature in enumerate(model.feature_names):
                # root style binning
                binning     = model.plot_options[feature]['binning']
                # linspace binning
                lin_binning[feature] = np.linspace(binning[1], binning[2], binning[0]+1)
                #digitize feature
                binned      = np.digitize(test_features[:,i_feature], lin_binning[feature] )
                # for each digit, create a mask to select the corresponding event in the bin (e.g. test_features[mask[0]] selects features in the first bin
                mask        = np.transpose( binned.reshape(-1,1)==range(1,len(lin_binning[feature])) )

                h_w0[feature]           = np.array([  w0[m].sum() for m in mask])
                h_derivative_prediction = np.array([ (w0.reshape(-1,1)*test_predictions)[m].sum(axis=0) for m in mask])
                h_derivative_truth      = np.array([ (np.transpose(np.array([test_weights[der] for der in bit.derivatives])))[m].sum(axis=0) for m in mask])

                h_ratio_prediction[feature] = h_derivative_prediction/h_w0[feature].reshape(-1,1) 
                h_ratio_truth[feature]      = h_derivative_truth/h_w0[feature].reshape(-1,1)

            n_pads = len(model.feature_names)+1
            n_col  = min(4, n_pads)
            n_rows = n_pads/n_col
            if n_rows*n_col<n_pads: n_rows+=1

            for logY in [False, True]:
                c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
                c1.Divide(n_col,n_rows)

                l = ROOT.TLegend(0.2,0.1,0.9,0.85)
                stuff.append(l)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)

                for i_feature, feature in enumerate(model.feature_names):

                    th1d_yield       = helpers.make_TH1F( (h_w0[feature], lin_binning[feature]) )
                    c1.cd(i_feature+1)
                    ROOT.gStyle.SetOptStat(0)
                    th1d_ratio_pred  = { der: helpers.make_TH1F( (h_ratio_prediction[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( bit.derivatives ) }
                    th1d_ratio_truth = { der: helpers.make_TH1F( (h_ratio_truth[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( bit.derivatives ) }
                    stuff.append(th1d_yield)
                    stuff.append(th1d_ratio_truth)
                    stuff.append(th1d_ratio_pred)

                    th1d_yield.SetLineColor(ROOT.kGray+2)
                    th1d_yield.SetMarkerColor(ROOT.kGray+2)
                    th1d_yield.SetMarkerStyle(0)
                    th1d_yield.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                    th1d_yield.SetTitle("")

                    th1d_yield.Draw("hist")

                    for i_der, der in enumerate(bit.derivatives):
                        th1d_ratio_truth[der].SetTitle("")
                        th1d_ratio_truth[der].SetLineColor(color[der])
                        th1d_ratio_truth[der].SetMarkerColor(color[der])
                        th1d_ratio_truth[der].SetMarkerStyle(0)
                        th1d_ratio_truth[der].SetLineWidth(2)
                        th1d_ratio_truth[der].SetLineStyle(ROOT.kDashed)
                        th1d_ratio_truth[der].GetXaxis().SetTitle(model.plot_options[feature]['tex'])

                        th1d_ratio_pred[der].SetTitle("")
                        th1d_ratio_pred[der].SetLineColor(color[der])
                        th1d_ratio_pred[der].SetMarkerColor(color[der])
                        th1d_ratio_pred[der].SetMarkerStyle(0)
                        th1d_ratio_pred[der].SetLineWidth(2)
                        th1d_ratio_pred[der].GetXaxis().SetTitle(model.plot_options[feature]['tex'])

                        tex_name = "_{%s}"%(",".join([model.tex[c].lstrip("C_{")[:-1] for c in der]))

                        if i_feature==0:
                            l.AddEntry( th1d_ratio_truth[der], "R"+tex_name)
                            l.AddEntry( th1d_ratio_pred[der],  "#hat{R}"+tex_name)

                    if i_feature==0:
                        l.AddEntry( th1d_yield, "yield (SM)")

                    max_ = max( map( lambda h:h.GetMaximum(), th1d_ratio_truth.values() ))
                    th1d_yield.Scale(max_/th1d_yield.GetMaximum())
                    th1d_yield   .Draw("hist")
                    ROOT.gPad.SetLogy(logY)
                    th1d_yield   .GetYaxis().SetRangeUser(0.1 if logY else 0, 10**(1.5)*max_ if logY else 1.5*max_)
                    th1d_yield   .Draw("hist")
                    for h in th1d_ratio_truth.values()+th1d_ratio_pred.values():
                        h .Draw("hsame")

                c1.cd(len(model.feature_names)+1)
                l.Draw()

                lines = [
                        (0.29, 0.9, 'N_{B} =%5i'%( max_n_tree ))
                        ]
                drawObjects = [ tex.DrawLatex(*line) for line in lines ]
                for o in drawObjects:
                    o.Draw()

                if not os.path.isdir(plot_directory_):
                    try:
                        os.makedirs( plot_directory_ )
                    except IOError:
                        pass
                from RootTools.plot.helpers import copyIndexPHP
                copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, "epoch_%05i.png"%(max_n_tree) ) )
                syncer.makeRemoteGif(plot_directory_, pattern="epoch_*.png", name="epoch" )
            syncer.sync()
