import ROOT
import os, sys
import argparse
import array
import numpy as np
import random
import cProfile
import time
from math import log, exp, sin, cos, sqrt, pi
import copy
from   RootTools.core.Sample import Sample
from   RootTools.core.standard          import *
import Analysis.Tools.syncer            as syncer
from   BoostedInformationTree           import BoostedInformationTree
import TMB.Tools.helpers                as helpers
from   Analysis.Tools                   import u_float
from   TMB.Tools.delphesCutInterpreter  import cutInterpreter
import pandas as pd
import uproot
import awkward

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',       action='store',      default='INFO',         nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'],             help="Log level for logging")
argParser.add_argument("--lumi",               action='store',      type=float,             default=137, help='Which lumi?')
argParser.add_argument('--output_directory',   action='store', type=str,   default=os.path.expandvars('/mnt/hephy/cms/$USER/BIT/'))
argParser.add_argument('--name',               action='store', type=str,   default='v1', help="Name of the training")
argParser.add_argument('--input_name',         action='store', type=str,   default='v1', help="Name of input directory for bit")
argParser.add_argument("--nTraining",          action="store",      default=500000,        type=int,                                         help="number of training events")
argParser.add_argument("--derivative",         action="store",      default=None,          nargs="*",                                        help="Maximum number of splits in node split")
argParser.add_argument('--overwrite',          action='store_true', help="Overwrite output?")
argParser.add_argument('--nSamples',           action='store', default=1, type=int, help="Number of produced training samples")
args = argParser.parse_args()

import toy_models.WZ as model

import TMB.Tools.user as user
plot_directory      = os.path.join( user. plot_directory, 'MVA', 'WZ', args.input_name)

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

Plot.setDefaults()
training_features,_ = model.getEvents(args.nTraining)
training_weights = model.getWeights(training_features, eft=model.default_eft_parameters)
if args.derivative is not None:
    for key in training_weights.keys():
        if key not in ( tuple(), tuple(args.derivative) ):
            del training_weights[key]

print "nEvents: %i Weights: %s" %( len(training_features), [ k for k in training_weights.keys() if k!=tuple()] )

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1)

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def PoissonLL( lam, n ):
    return lam - n*np.nan_to_num(np.log(lam)) + sum( [log(n_) for n_ in range(1,n+1)] ) #edited

lumi = args.lumi
#Boosting
n_trees       = 150
max_depth     = 4
learning_rate = 0.20
min_size      = 50
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

plot_directory      = os.path.join( user. plot_directory, 'MVA', 'WZ', args.name)

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

training_features,_ = model.getEvents(args.nTraining)
training_weights = model.getWeights(training_features, eft=model.default_eft_parameters)
training_samples = np.array([[training_features,training_weights]])
bit_predictions_train = { key:bits[key].vectorized_predict(training_features) for key in  training_weights.keys() if key!=tuple() }
bit = np.array([bit_predictions_train])
for i in range(1,args.nSamples):
    training_features,_ = model.getEvents(args.nTraining)
    training_weights = model.getWeights(training_features, eft=model.default_eft_parameters)
    np.append(training_samples, [training_features,training_weights])
    bit_predictions_train = { key:bits[key].vectorized_predict(training_features) for key in  training_weights.keys() if key!=tuple() }
    np.append(bit,bit_predictions_train)
#print training_samples
#print "#################"
#print training_samples[0][1][('cW','cW')]
#print "#################"
#print training_samples[0][1]
colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kRed ]
def make_TH1F( h ):
    vals, thrs = h
    histo = ROOT.TH1F("h","h",len(vals),0,len(vals))
    for i_v, v in enumerate(vals):
        histo.SetBinContent(i_v+1, v)
    return histo

binning_quantiles = [.01, .025, .05, .1, .2, .3, .5, .6, .7, .8, .9, .95, .975, .99,]
#WC = 'cW'
WC = 'c3PQ'

WC_vals = [i/20. for i in range(-10,11)]
qs = {'linear'   :{'color':ROOT.kGreen},
                'quadratic':{'color':ROOT.kRed},
                'total'    :{'color':ROOT.kBlack},
                'true'     :{'color':ROOT.kBlue}
                }
for q_name, q in qs.iteritems():
    q['nll_tGraph'] = ROOT.TGraph( len(WC_vals) )
    q['nll_tGraph'].SetLineColor( q['color'] )
    q['nll_tGraph'].SetMarkerColor( q['color'] )
    q['nll_tGraph'].SetMarkerStyle( 0 )
    q['nll_tGraph'].SetLineWidth(2)
    q['unbinned_nll_tGraph'] = ROOT.TGraph( len(WC_vals) )
    q['unbinned_nll_tGraph'].SetLineColor( q['color'] )
    q['unbinned_nll_tGraph'].SetMarkerColor( q['color'] )
    q['unbinned_nll_tGraph'].SetMarkerStyle( 0 )
    q['unbinned_nll_tGraph'].SetLineWidth(2)

    q['val'] = {}
    q['nll'] = {}
    q['poissonPrefitNLL'] = {}

for i_WC_val, WC_val in enumerate(WC_vals):
    w_sm = {}
    w_bsm = {}
    q_lin = {}
    q_tot = {}
    q_quad = {}

    for i_training_sample, training_sample in enumerate(training_samples):
        #print i_training_sample, training_sample
        w_sm[i_training_sample] = training_sample[1][()]
        #w_bsm["sample_%s"% ('_'.join([str(i_training_sample)]))] = training_sample[1][()]+WC_val*training_sample[1][(WC,)] + 0.5*WC_val**2 * training_sample[1][(WC,WC)]
        w_bsm[i_training_sample] = training_sample[1][()]+WC_val*training_sample[1][(WC,)] + 0.5*WC_val**2 * training_sample[1][(WC,WC)]
        qs['total']['val'][i_training_sample] = bit[i_training_sample][(WC,)] + 0.5 * WC_val*bit[i_training_sample][(WC,WC)]
        qs['true']['val'][i_training_sample] = w_bsm[i_training_sample]/w_sm[i_training_sample]
        qs['linear']['val'][i_training_sample] = bit[i_training_sample][(WC,)]
        qs['quadratic']['val'][i_training_sample] = bit[i_training_sample][(WC,WC)]

        if i_training_sample == 0:
            for q_name, q in qs.iteritems():
                weighted_binning = [-float('inf')]+list(weighted_quantile( q['val'][i_training_sample], binning_quantiles, w_sm[i_training_sample]))+[float('inf')]
                q['binning'] = helpers.remove_duplicates( weighted_binning )
                q['observation'] = [0]*( len(q['binning'])-1 )
                q['h_SM']      = {}
                q['h_BSM']     = {}
                q['histo_SM']  = {}
                q['histo_BSM'] = {}

        qs['total']['binning'] = qs['linear']['binning']
        for q_name, q in qs.iteritems():
            q['h_SM'][i_training_sample]      = np.histogram(q['val'][i_training_sample], q['binning'], weights = w_sm[i_training_sample]*float(lumi))#/config.scale_weight)
            q['h_BSM'][i_training_sample]      = np.histogram(q['val'][i_training_sample], q['binning'], weights = w_bsm[i_training_sample]*float(lumi))#/config.scale_weight)
            q['histo_SM'] [i_training_sample] = make_TH1F(q['h_SM'][i_training_sample])
            q['histo_BSM'] [i_training_sample] = make_TH1F(q['h_BSM'][i_training_sample])

    for q_name, q in qs.iteritems():
        q['nll'][WC_val] = 0.
        for i_b in range(len(q['binning'])-1):
            expectations = [ q['h_BSM'][i_training_sample][0][i_b] for training_sample in training_samples ]
            observations = [ q['h_SM'][i_training_sample][0][i_b] for training_sample in training_samples ]
            q['nll'][WC_val] += PoissonLL( sum(expectations), int(round(sum(observations))) )
        q['nll_tGraph'].SetPoint( i_WC_val, WC_val, q['nll'][WC_val] )
        print "nll",q_name, "WC_val",WC_val,":",q['nll'][WC_val]
        for i_training_sample, training_sample in enumerate(training_samples):
            q['histo_SM'] [i_training_sample].style      = styles.lineStyle(colors[i_training_sample], dashed = True )
            q['histo_SM'] [i_training_sample].legendText = i_training_sample #training_sample.name#+" (SM)"
            q['histo_BSM'][i_training_sample].style      = styles.lineStyle(colors[i_training_sample], dashed = False )
            q['histo_BSM'][i_training_sample].legendText = i_training_sample #training_sample.name#+" (cHW=%4.3f)"%cHW

        plot = Plot.fromHisto(name = "q_%s_%s_%4.3f"%(q_name,WC,WC_val),
                    histos = [[q['histo_SM'] [i_s] for i_s, s in enumerate(training_samples)],
                            [ q['histo_BSM'][i_s] for i_s, s in enumerate(training_samples)],],
                    texX = "q_{%s=%4.3f}"%(WC,WC_val) , texY = "Number of events" )
        for log_ in [True]:
            plot_directory_ = os.path.join(plot_directory, ("log" if log_ else "lin"))
            plotting.draw(plot, plot_directory = plot_directory_, ratio = {'histos':[(1,0)], 'texY': 'Ratio'}, logY = log_, logX = False, yRange = (10**-2,"auto"),
                        legend = ([0.20,0.75,0.9,0.88],3), copyIndexPHP=True, )

    unbinned_nll_tot  = 0
    unbinned_nll_true  = 0
    unbinned_nll_quad  = 0
    unbinned_nll_lin  = 0
    total_xsec_bsm_true    = 0
    total_xsec_bsm_tot    = 0
    total_xsec_bsm_quad    = 0
    total_xsec_bsm_lin    = 0
    total_xsec_sm     = 0
    rescale           = float(lumi)#/config.scale_weight
    for i_training_sample, training_sample in enumerate(training_samples):
        unbinned_nll_tot  += -rescale*np.sum(np.nan_to_num(w_sm[i_training_sample]*(np.log(1+WC_val*bit[i_training_sample][(WC,)] + 0.5*WC_val**2*bit[i_training_sample][(WC,WC)] ))))
        unbinned_nll_true  += -rescale*np.sum(np.nan_to_num(w_sm[i_training_sample]*(np.log(w_bsm[i_training_sample]/w_sm[i_training_sample]))))
        unbinned_nll_quad += -rescale*np.sum(np.nan_to_num(w_sm[i_training_sample]*(np.log(1+0.5*WC_val**2*bit[i_training_sample][(WC,WC)] ))))
        unbinned_nll_lin  += -rescale*np.sum(np.nan_to_num(w_sm[i_training_sample]*(np.log(1+WC_val*bit[i_training_sample][(WC,)] ))))
        total_xsec_bsm_true  += rescale*sum(w_bsm[i_training_sample])
        total_xsec_bsm_tot   += rescale*sum((1+WC_val*bit[i_training_sample][(WC,)] + 0.5*WC_val**2*bit[i_training_sample][(WC,WC)])*w_sm[i_training_sample])
        total_xsec_bsm_quad  += rescale*sum((1+ 0.5*WC_val**2*bit[i_training_sample][(WC,WC)])*w_sm[i_training_sample])
        total_xsec_bsm_lin   += rescale*sum((1+WC_val*bit[i_training_sample][(WC,)])*w_sm[i_training_sample])
        total_xsec_sm     += rescale*sum(w_sm[i_training_sample])

    Poissonian_NLL_term_true      = -total_xsec_sm*log( total_xsec_bsm_true/total_xsec_sm )
    Poissonian_NLL_term_tot      = -total_xsec_sm*log( total_xsec_bsm_tot/total_xsec_sm )
    Poissonian_NLL_term_quad      = -total_xsec_sm*log( total_xsec_bsm_quad/total_xsec_sm )
    Poissonian_NLL_term_lin      = -total_xsec_sm*log( total_xsec_bsm_lin/total_xsec_sm )
    qs['total']     ['unbinned_nll_tGraph'].SetPoint( i_WC_val, WC_val, unbinned_nll_tot - Poissonian_NLL_term_tot )
    qs['true']     ['unbinned_nll_tGraph'].SetPoint( i_WC_val, WC_val, unbinned_nll_true - Poissonian_NLL_term_true )
    qs['quadratic'] ['unbinned_nll_tGraph'].SetPoint( i_WC_val, WC_val, unbinned_nll_quad- Poissonian_NLL_term_quad )
    qs['linear']    ['unbinned_nll_tGraph'].SetPoint( i_WC_val, WC_val, unbinned_nll_lin - Poissonian_NLL_term_lin )

for key in  [ 'nll_tGraph', 'unbinned_nll_tGraph']:
    c1 = ROOT.TCanvas()
    ROOT.gStyle.SetOptStat(0)
    c1.SetTitle("")
    l = ROOT.TLegend(0.55, 0.8, 0.8, 0.9)
    l.SetFillStyle(0)
    l.SetShadowColor(ROOT.kWhite)
    l.SetBorderSize(0)
    first = True
    for q_name, q in qs.iteritems():
        l.AddEntry( q[key], q_name )
        q[key].Draw("AL" if first else "L")
        q[key].SetTitle("")
        q[key].GetXaxis().SetTitle(WC)
        q[key].GetYaxis().SetTitle("NLL")
        first = False

    l.Draw()
    c1.RedrawAxis()
    c1.Print(os.path.join(plot_directory, key+".png"))
    c1.Print(os.path.join(plot_directory, key+".pdf"))
    c1.Print(os.path.join(plot_directory, key+".root"))

syncer.sync()





