#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
sys.path.insert( 0, ".." )
from math import log, exp, sin, cos, sqrt, pi
import copy
import operator
import itertools

# RootTools
from RootTools.core.standard   import *

# Analysis
import Analysis.Tools.syncer as syncer

# BIT
from BoostedInformationTree import BoostedInformationTree

# User
import user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="mBIT_ZH",       help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="ZH_Nakamura",   help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default=None, type=str,  help="prefix")
argParser.add_argument("--nTraining",          action="store",      default=50000,        type=int,  help="number of training events")
argParser.add_argument("--coefficients",       action="store",      default=['cHW', 'cHWtil', 'cHQ3'],  nargs="*", help="Maximum number of splits in node split")
argParser.add_argument('--overwrite',          action='store_true', help="Overwrite output?")
argParser.add_argument('--debug',              action='store_true', help="Make debug plots?")
argParser.add_argument('--positive',           action='store_true', help="positive?")
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

#for derivative in model.derivatives:
#    if derivative != tuple():
#        model.multi_bit_cfg[derivative].update( extra_args )

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

features = model.getEvents(args.nTraining)
training_weights  = model.getWeights(features, eft=model.default_eft_parameters)
print ("Created training data set of size %i" % len(features) )

# reduce training weights to the dimensions we learn
args.coefficients = sorted(args.coefficients)

for key in training_weights.keys():
    if key==tuple(): continue
    if not all( [ k in args.coefficients for k in key] ): 
        del training_weights[key]

model.first_derivatives  = filter( lambda der: all( [ c in args.coefficients for c in der ] ), model.first_derivatives )
model.second_derivatives = filter( lambda der: all( [ c in args.coefficients for c in der ] ), model.second_derivatives )
model.derivatives = [tuple()] + model.first_derivatives + model.second_derivatives

print "nEvents: %i Weights: %s" %( len(features), [ k for k in training_weights.keys() if k!=tuple()] )

# cfg & preparation for node split
min_size    = 50
max_n_split = -1
min_node_size_neg_adjust = True
size        = len(features)
training_weights = np.array([training_weights[der] for der in model.derivatives]).transpose()

base_points = []
for comb in list(itertools.combinations_with_replacement(args.coefficients,1))+list(itertools.combinations_with_replacement(args.coefficients,2)):
    base_points.append( {c:comb.count(c) for c in args.coefficients} )

base_point_const = np.array([[ reduce(operator.mul, [point[coeff] if point.has_key(coeff) else 0 for coeff in der ], 1) for der in model.derivatives] for point in base_points]).astype('float')
for i_der, der in enumerate(model.derivatives):
    if not (len(der)==2 and der[0]==der[1]): continue 
    for i_point in range(len(base_points)):
        #print i_point, i_der, der, base_point_const[i_point][i_der], base_point_const[i_point][i_der]/2.
        base_point_const[i_point][i_der] = base_point_const[i_point][i_der]/2.

const = np.zeros((1,len(model.derivatives)))
const[0,0]=1
base_point_const_ = np.concatenate((const, base_point_const)) 

# get_split_vectorized
split_i_feature, split_value, split_gain, split_left_group = 0, -float('inf'), 0, None

for i_feature in range(len(features[0])):
    print "i_feature", i_feature
    feature_values = features[:,i_feature]

    feature_sorted_indices = np.argsort(feature_values)

    sorted_weight_sums     = np.cumsum(training_weights[feature_sorted_indices],axis=0) # FIXME cumsum does not respect max_n_split

    # respect min size for split
    if max_n_split<2:
        plateau_and_split_range_mask = np.ones(size-1, dtype=np.dtype('bool'))
    else:
        min_, max_ = min(feature_values), max(feature_values)
        #print "_depth",self._depth, "len(feature_values)",len(feature_values), "min_, max_", min_, max_
        plateau_and_split_range_mask  = np.digitize(feature_values[feature_sorted_indices], np.arange (min_, max_, (max_-min_)/(max_n_split+1)))
        #print len(plateau_and_split_range_mask), plateau_and_split_range_mask
        plateau_and_split_range_mask = plateau_and_split_range_mask[1:]-plateau_and_split_range_mask[:-1]
        plateau_and_split_range_mask = np.insert( plateau_and_split_range_mask, 0, 0).astype('bool')[:-1]
        #print "plateau_and_split_range_mask", plateau_and_split_range_mask
        #print "CUTS", feature_values[feature_sorted_indices][:-1][plateau_and_split_range_mask] 

    if min_size > 1:
        plateau_and_split_range_mask[0:min_size-1] = False
        plateau_and_split_range_mask[-min_size+1:] = False
    plateau_and_split_range_mask &= (np.diff(feature_values[feature_sorted_indices]) != 0)

    total_weight_sum         = sorted_weight_sums[-1]
    sorted_weight_sums       = sorted_weight_sums[0:-1]
    sorted_weight_sums_right = total_weight_sum-sorted_weight_sums

    if args.positive: # test positivity
        pos       = np.apply_along_axis(all, 1, np.dot(sorted_weight_sums,base_point_const_.transpose())>=0)
        pos_right = np.apply_along_axis(all, 1, np.dot(sorted_weight_sums_right,base_point_const_.transpose())>=0) 

    # Never allow negative yields
    plateau_and_split_range_mask &= (sorted_weight_sums[:,0]>0)
    plateau_and_split_range_mask &= (sorted_weight_sums_right[:,0]>0)

    neg_loss_gains = np.sum(np.dot( sorted_weight_sums, base_point_const.transpose())**2,axis=1)/sorted_weight_sums[:,0]
    neg_loss_gains+= np.sum(np.dot( sorted_weight_sums_right, base_point_const.transpose())**2,axis=1)/sorted_weight_sums_right[:,0]

    with np.errstate(divide='ignore'):
        if min_node_size_neg_adjust>0:
            # Defining the negative weight fraction as f=n^+/n^- the relative statistical MC uncertainty in a yield is 1/sqrt(n) sqrt(1+f)/sqrt(1-f). 
            # The min node size should therefore be increased to (1+f)/(1-f)
            sorted_pos_sums       = np.cumsum( (training_weights[:,0]>0).astype('int')[feature_sorted_indices])
            total_pos_sum         = sorted_pos_sums[-1]
            sorted_pos_sums       = sorted_pos_sums[0:-1]
            sorted_pos_sums_right = total_pos_sum-sorted_pos_sums
            sorted_neg_sum       = np.array(range(1, len(training_weights[:,0])))     - sorted_pos_sums
            sorted_neg_sum_right = np.array(range(len(training_weights[:,0])-1,0,-1)) - sorted_pos_sums_right
            f      = sorted_neg_sum/sorted_pos_sums.astype(float)
            f_right= sorted_neg_sum_right/sorted_pos_sums_right.astype(float)

            plateau_and_split_range_mask &= range(1, len(training_weights[:,0]))>(1+f)/(1-f)*min_size
            plateau_and_split_range_mask &= np.array(range(len(training_weights[:,0])-1,0,-1)) >(1+f_right)/(1-f_right)*min_size

    plateau_and_split_range_mask = plateau_and_split_range_mask.astype(int)

    gain_masked = np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask
    argmax_fi   = np.argmax(gain_masked)
    gain        = gain_masked[argmax_fi]

    value = feature_values[feature_sorted_indices[argmax_fi]]

    if gain > split_gain:
        split_i_feature = i_feature
        split_value     = value
        split_gain      = gain

assert not np.isnan(split_value)

print split_i_feature, split_value, split_gain
split_left_group = features[:,split_i_feature]<=split_value if not  np.isnan(split_value) else np.ones(size, dtype='bool')
