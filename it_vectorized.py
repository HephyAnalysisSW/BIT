#!/usr/bin/env python
# Standard imports
import numpy as np
import copy
import cProfile
import operator 
import time

from Analysis.Tools.helpers import chunk

class Node:
    def __init__( self, features, weights, FI_func, max_depth, min_size, sorted_feature_values_and_weight_sums, weight_mask, diff_weight_mask, split_method="python_loop", depth=0):

        ## basic BDT configuration
        self.max_depth  = max_depth
        self.min_size   = min_size

        # data set
        self.features   = features
        self.weights    = weights
       
        # FI func
        self.FI_func    = FI_func
 
        assert len(self.features) == len(self.weights), "Unequal length!"

        # weight mask to calculate FI vectorized
        self.polynomial_dim = self.weights.shape[-1]
        self.sorted_feature_values_and_weight_sums = sorted_feature_values_and_weight_sums 
        self.weight_mask = weight_mask
        self.diff_weight_mask = diff_weight_mask
        assert len(self.weight_mask) == self.polynomial_dim, "Unequal length weights and mask!"
        assert len(self.diff_weight_mask) == self.polynomial_dim, "Unequal length weights and diff mask!"

        self.size       = len(self.features)

        # keep track of recursion depth
        self.depth      = depth
        self.split_method = split_method

        self.split(depth=depth)

    # compute the total FI from a set of booleans defining the 'left' box and (by negation) the 'right' box
    def FI_from_group( self, group):
        ''' Calculate FI for selection
        '''
        sum_    =  sum(self.weights[group])
        if type(sum_)==int and sum_==0: return 0 # the case where we get a list of False
        return self.FI_func(sum_)

    # convinience for debugging
    def FI_threshold_lower( self, i_feature, value):
        feature_values = self.features[:,i_feature]
        # get column & loop over all values
        group = feature_values<value
        return self.FI_from_group( group )

    # convinience for debugging
    def FI_threshold_higher( self, i_feature, value):
        feature_values = self.features[:,i_feature]
        # get column & loop over all values
        group = feature_values>=value
        return self.FI_from_group( group )
         
    def get_split( self ):
        ''' determine where to split the features
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_score, self.split_left_group = None, float('nan'), 0, None

        # loop over features
        for i_feature in range(len(self.features[0])):
            feature_values = self.features[:,i_feature]
            # get column & loop over all values
            for value in feature_values:
                left_group = feature_values<value
                DFI =  self.FI_from_group(  left_group )
                DFI += self.FI_from_group( ~left_group )
                if DFI > self.split_score:
                    self.split_i_feature = i_feature
                    self.split_value     = value
                    self.split_score     = DFI
                    self.split_left_group= left_group

        #print ("final:get_split", self.split_i_feature, self.split_value)
        #return {'split_i_feature':split_i_feature, 'split_value':split_value, 'split_score':split_score, 'split_left_group':split_left_group}

    def get_split_fast( self ):
        ''' determine where to split the features
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_score, self.split_left_group = None, float('nan'), 0, None

        # loop over features
        for i_feature in range(len(self.features[0])):
            feature_values = self.features[:,i_feature]

            weight_sum = np.zeros(len(self.weights[0]))
            weight_sums= []
            for position, value in sorted(enumerate(feature_values), key=operator.itemgetter(1)):
                weight_sum = weight_sum+self.weights[position]
                weight_sums.append( (value,  weight_sum) )

            total_weights = weight_sums[-1][1]
            for value, weight_sum in weight_sums:
                #print weight_sum, total_weights-weight_sum
                score = self.FI_func(weight_sum) + self.FI_func( total_weights-weight_sum )
                if score > self.split_score: 
                    self.split_i_feature = i_feature
                    self.split_value     = value
                    self.split_score     = score

        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value
 
    def get_split_vectorized( self ):
        ''' determine where to split the features, first vectorized version of FI maximization
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_score, self.split_left_group = None, float('nan'), 0, None

        # loop over features
        for i_feature in range(len(self.features[0])):
            feature_values = self.features[:,i_feature]
        
            if self.split_method == 'vectorized_split_and_weight_sums':
                feature_sorted_indices = np.argsort(feature_values)
                weight_sums = np.cumsum(self.weights[feature_sorted_indices], axis=0)

                #tic = time.time()
                idx, score = self.find_split_vectorized(weight_sums)
                #toc = time.time()
                #print("vectorized split in {time:0.4f} seconds".format(time=toc-tic))
                value = feature_values[feature_sorted_indices[idx]]
                #value = self.sorted_feature_values_and_weight_sums[i_feature]['sorted_feature_values'][idx]
            else:
                weight_sum = np.zeros(len(self.weights[0]))
                weight_sums= []
                pure_weight_sums = []
                #tic = time.time()
                for position, value in sorted(enumerate(feature_values), key=operator.itemgetter(1)):
                    weight_sum = weight_sum+self.weights[position]
                    weight_sums.append( (value,  weight_sum) )
                    pure_weight_sums.append(weight_sum)
                #toc = time.time()
                #print("sorting/summing in {time:0.4f} seconds".format(time=toc-tic))
                idx, score = self.find_split_vectorized(np.array(pure_weight_sums))
                value = weight_sums[idx][0]

            if score > self.split_score: 
                self.split_i_feature = i_feature
                self.split_value     = value
                self.split_score     = score

        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value

    def find_split_vectorized(self, sorted_weight_sums):
        total_weight_sum = sorted_weight_sums[-1]
        sorted_weight_sums = sorted_weight_sums[0:-1]
        yields_left = np.sum(sorted_weight_sums*self.weight_mask, axis=1)
        yields_right = np.sum((total_weight_sum-sorted_weight_sums)*self.weight_mask,axis=1)
        diff_yields_left = sorted_weight_sums*self.diff_weight_mask
        diff_yields_right = (total_weight_sum-sorted_weight_sums)*self.diff_weight_mask
        fisher_information_left = np.sum(diff_yields_left, axis=1)**2/yields_left
        fisher_information_right = np.sum(diff_yields_right, axis=1)**2/yields_right
        fisher_scores = fisher_information_left + fisher_information_right
        argmax_fi = np.argmax(np.nan_to_num(fisher_scores))
        return argmax_fi, fisher_scores[argmax_fi]

    # Create child splits for a node or make terminal
    def split(self, depth):

        # Find the best split
        #tic = time.time()
        #self.get_split()
        #self.get_split_fast()
        if self.split_method == "python_loop":
            self.get_split_fast()
        elif self.split_method == "vectorized_split" or self.split_method == "vectorized_split_and_weight_sums":
            self.get_split_vectorized()
        else:
            raise ValueError("no such split method %s" % self.split_method)
        #toc = time.time()

        #print("get_split in {time:0.4f} seconds".format(time=toc-tic))


        # check for max depth or a 'no' split
        if  self.max_depth <= depth+1 or (not any(self.split_left_group)) or all(self.split_left_group): # Jason Brownlee starts counting depth at 1, we start counting at 0, hence the +1
            #print ("Choice2", depth, self.FI_from_group(self.split_left_group), self.FI_from_group(~self.split_left_group) )
            # The split was good, but we stop splitting further. Put the result of the split in the left/right boxes.
            self.left, self.right = ResultNode(self.FI_from_group(self.split_left_group), np.count_nonzero(self.split_left_group)), ResultNode(self.FI_from_group(~self.split_left_group), np.count_nonzero(~self.split_left_group))
            return
        # process left child
        if np.count_nonzero(self.split_left_group) <= min_size:
            #print ("Choice3", depth, self.FI_from_group(self.split_left_group) )
            # Too few events in the left box. We stop.
            self.left             = ResultNode(self.FI_from_group(self.split_left_group), np.count_nonzero(self.split_left_group))
        else:
            #print ("Choice4", depth )
            # Continue splitting left box. 
            self.left             = Node(self.features[self.split_left_group], self.weights[self.split_left_group], FI_func=self.FI_func, max_depth=self.max_depth, min_size=self.min_size, sorted_feature_values_and_weight_sums=self.sorted_feature_values_and_weight_sums, weight_mask=self.weight_mask, diff_weight_mask=self.diff_weight_mask, split_method=self.split_method, depth=self.depth+1 )
        # process right child
        if np.count_nonzero(~self.split_left_group) <= min_size:
            #print ("Choice5", depth, self.FI_from_group(~self.split_left_group) )
            # Too few events in the right box. We stop.
            self.right            = ResultNode(self.FI_from_group(~self.split_left_group), np.count_nonzero(~self.split_left_group))
        else:
            #print ("Choice6", depth  )
            # Continue splitting right box. 
            self.right            = Node(self.features[~self.split_left_group], self.weights[~self.split_left_group], FI_func=self.FI_func, max_depth=self.max_depth, min_size=self.min_size, sorted_feature_values_and_weight_sums=self.sorted_feature_values_and_weight_sums, weight_mask=self.weight_mask, diff_weight_mask=self.diff_weight_mask, split_method=self.split_method, depth=self.depth+1 )

#    # Prediction    
#    def predict( self, row ):
#        ''' obtain the result by recursively descending down the tree
#        '''
#        node = self.left if row[self.split_i_feature]<self.split_value else self.right
#        if isinstance(node, ResultNode):
#            return node.return_value
#        else:
#            return node.predict(row)

    # Print a decision tree
    def print_tree(self, depth=0):
        print('%s[X%d <= %.3f]' % ((self.depth*' ', self.split_i_feature, self.split_value)))
        for node in [self.left, self.right]:
            node.print_tree(depth+1)

    def total_FI(self):
        result = 0
        for node in [self.left, self.right]:
            result += node.return_value if isinstance(node, ResultNode) else node.total_FI()
        return result

class ResultNode:
    ''' Simple helper class to store result value.
    '''
    def __init__( self, return_value, size ):
        self.return_value   = return_value
        self.size = size
    def print_tree(self, depth=0):
        print('%s[%s] (%d)' % (((depth)*' ', self.return_value, self.size)))

import uproot
import awkward
import numpy as np
import pandas as pd

# Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--maxEvents', action='store', type=int, default=10000)
argParser.add_argument('--minDepth', action='store', type=int, default=1)
argParser.add_argument('--maxDepth', action='store', type=int, default=4)
argParser.add_argument('--splitMethod', action='store', type=str, default='vectorized_split_and_weight_sums')
args = argParser.parse_args()

max_events  = args.maxEvents
input_file  = "/eos/vbc/user/robert.schoefbeck/TMB/bit/MVA-training/ttG_WG_small/WGToLNu_fast/WGToLNu_fast.root"
upfile      = uproot.open( input_file )
tree        = upfile["Events"]
n_events    = len( upfile["Events"] )
n_events    = min(max_events, n_events)
entrystart, entrystop = 0, n_events 

# Load features
#branches    = [ "mva_photon_pt", ]#"mva_photon_eta", "mva_photonJetdR", "mva_photonLepdR", "mva_mT" ]
branches    = [ "mva_photon_pt" ]#, "mva_photon_eta", "mva_photonJetdR", "mva_photonLepdR", "mva_mT" ]
df          = tree.pandas.df(branches = branches, entrystart=entrystart, entrystop=entrystop)
features    = df.values

print(features.shape)

# Load weights
#from Analysis.Tools.WeightInfo import WeightInfo
# custom WeightInfo
from WeightInfo import WeightInfo
w = WeightInfo("/eos/vbc/user/robert.schoefbeck/gridpacks/v6/WGToLNu_reweight_card.pkl")
w.set_order(2)

# Load all weights and reshape the array according to ndof from weightInfo
weights     = tree.pandas.df(branches = ["p_C"], entrystart=entrystart, entrystop=entrystop).values.reshape((-1,w.nid))
print(weights.shape)

min_size = 50

assert len(features)==len(weights), "Need equal length for weights and features."

FI_func = lambda coeffs: w.get_fisherInformation_matrix( coeffs, variables = ['cWWW'], cWWW=1)[1][0][0]
weight_mask = w.get_weight_mask( cWWW=1 )

diff_weight_mask = w.get_diff_mask( 'cWWW', cWWW=1 )

#TODO:
training_weights         = np.dot(weights, w.get_weight_mask(cWWW=1))
training_diff_weights    = np.dot(weights, w.get_diff_mask('cWWW', cWWW=1))

print("number of events %d" % len(features))
tic_overall = time.time()

sorted_feature_values_and_weight_sums = []

for max_depth in range(args.minDepth,args.maxDepth+1):
    tic  = time.time()
    node = Node( features, weights, FI_func=FI_func, max_depth=max_depth, min_size=min_size, sorted_feature_values_and_weight_sums=sorted_feature_values_and_weight_sums, weight_mask=weight_mask, diff_weight_mask=diff_weight_mask, split_method=args.splitMethod )
    toc = time.time()
    print("tree construction in {time:0.4f} seconds".format(time=toc-tic))
    print "max_depth", max_depth
    print 
    node.print_tree()
    print 
    print "Total FI", node.total_FI() 
    print 
    print 

toc_overall = time.time()
all_construction_time = toc_overall-tic_overall
print("all constructions in {time:0.4f} seconds".format(time=all_construction_time))
