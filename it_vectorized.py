#!/usr/bin/env python
# Standard imports
import numpy as np
import copy
import cProfile
import operator 
import time

from Analysis.Tools.helpers import chunk

class Node:
    def __init__( self, features, max_depth, min_size, training_weights, training_diff_weights, split_method="python_loop", depth=0):

        ## basic BDT configuration
        self.max_depth  = max_depth
        self.min_size   = min_size

        # data set
        self.features   = features

        self.training_weights      = training_weights
        self.training_diff_weights = training_diff_weights       
 
        assert len(self.features) == len(self.training_weights) == len(self.training_diff_weights), "Unequal length!"

        self.size       = len(self.features)

        # keep track of recursion depth
        self.depth      = depth
        self.split_method = split_method

        self.split(depth=depth)

    # compute the score from a set of booleans defining the 'left' box and (by negation) the 'right' box
    def score_from_group( self, group):
        ''' Calculate FI for selection
        '''
        sum_diff_ = sum(self.training_diff_weights[group])
        sum_      = sum(self.training_weights[group])

        if sum_==0.: return 0

        return sum_diff_/sum_ 

    # compute the total FI from a set of booleans defining the 'left' box and (by negation) the 'right' box
    def FI_from_group( self, group):
        ''' Calculate FI for selection
        '''
        sum_diff_ = sum(self.training_diff_weights[group])
        sum_      = sum(self.training_weights[group])

        if sum_==0.: return 0

        return sum_diff_**2/sum_ 

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

    def get_split_fast( self ):
        ''' determine where to split the features
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_gain, self.split_left_group = None, float('nan'), 0, None

        # loop over features
        for i_feature in range(len(self.features[0])):
            feature_values = self.features[:,i_feature]

            weight_sum = 0. 
            weight_sums= []
            weight_diff_sum = 0. 
            weight_diff_sums= []
            for position, value in sorted(enumerate(feature_values), key=operator.itemgetter(1)):
                weight_sum = weight_sum+self.training_weights[position]
                weight_sums.append( (value,  weight_sum) )
                weight_diff_sum = weight_diff_sum+self.training_diff_weights[position]
                weight_diff_sums.append(  weight_diff_sum )

            total_weight = weight_sums[-1][1]
            total_diff_weight = weight_diff_sums[-1]
            for i_value, (value, weight_sum) in enumerate(weight_sums):
                #print weight_sum, total_weights-weight_sum
                gain = weight_diff_sums[i_value]**2/weight_sum + (total_diff_weight-weight_diff_sums[i_value])**2/(total_weight-weight_sum) 
                if gain > self.split_gain: 
                    self.split_i_feature = i_feature
                    self.split_value     = value
                    self.split_gain     = gain

        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value
        #print "python_loop", self.split_i_feature, self.split_value,  self.split_left_group 
         
    def get_split_vectorized( self ):
        ''' determine where to split the features, first vectorized version of FI maximization
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_gain, self.split_left_group = None, float('nan'), 0, None

        # loop over features
        for i_feature in range(len(self.features[0])):
            feature_values = self.features[:,i_feature]
        
            feature_sorted_indices = np.argsort(feature_values)
            weight_sums      = np.cumsum(self.training_weights[feature_sorted_indices])
            weight_diff_sums = np.cumsum(self.training_diff_weights[feature_sorted_indices])

            #print weight_sums
            #print weight_diff_sums
            #tic = time.time()
            idx, gain = self.find_split_vectorized(weight_sums, weight_diff_sums)
            #toc = time.time()
            #print("vectorized split in {time:0.4f} seconds".format(time=toc-tic))
            value = feature_values[feature_sorted_indices[idx]]

            if gain > self.split_gain: 
                self.split_i_feature = i_feature
                self.split_value     = value
                self.split_gain     = gain

        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value

    def find_split_vectorized(self, sorted_weight_sums, sorted_weight_diff_sums):
        total_weight_sum         = sorted_weight_sums[-1]
        total_diff_weight_sum    = sorted_weight_diff_sums[-1]
        sorted_weight_sums       = sorted_weight_sums[0:-1]
        sorted_weight_diff_sums  = sorted_weight_diff_sums[0:-1]

        fisher_information_left  = sorted_weight_diff_sums*sorted_weight_diff_sums/sorted_weight_sums 
        fisher_information_right = (total_diff_weight_sum-sorted_weight_diff_sums)*(total_diff_weight_sum-sorted_weight_diff_sums)/(total_weight_sum-sorted_weight_sums) 

        fisher_gains = fisher_information_left + fisher_information_right
        argmax_fi = np.argmax(np.nan_to_num(fisher_gains))
        return argmax_fi, fisher_gains[argmax_fi]

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

        #print("get_split in {time:0.4f} seconds".format(time=toc-tic))

        # decide what we put in the result node
        result_funcs = { 
            'size':  lambda group: np.count_nonzero(group),
            'FI'  :  lambda group: self.FI_from_group(group),
            'score': lambda group: self.score_from_group(group)
            }

        # check for max depth or a 'no' split
        if  self.max_depth <= depth+1 or (not any(self.split_left_group)) or all(self.split_left_group): # Jason Brownlee starts counting depth at 1, we start counting at 0, hence the +1
            #print ("Choice2", depth, result_func(self.split_left_group), result_func(~self.split_left_group) )
            # The split was good, but we stop splitting further. Put the result of the split in the left/right boxes.
            self.left, self.right = ResultNode(**{val:func(self.split_left_group) for val, func in result_funcs.iteritems()}), ResultNode(**{val:func(~self.split_left_group) for val, func in result_funcs.iteritems()})
            return
        # process left child
        if np.count_nonzero(self.split_left_group) <= min_size:
            #print ("Choice3", depth, result_func(self.split_left_group) )
            # Too few events in the left box. We stop.
            self.left             = ResultNode(**{val:func(self.split_left_group) for val, func in result_funcs.iteritems()})
        else:
            #print ("Choice4", depth )
            # Continue splitting left box. 
            self.left             = Node(self.features[self.split_left_group], max_depth=self.max_depth, min_size=self.min_size, training_weights = self.training_weights[self.split_left_group], training_diff_weights = self.training_diff_weights[self.split_left_group], split_method=self.split_method, depth=self.depth+1 )
        # process right child
        if np.count_nonzero(~self.split_left_group) <= min_size:
            #print ("Choice5", depth, result_func(~self.split_left_group) )
            # Too few events in the right box. We stop.
            self.right            = ResultNode(**{val:func(~self.split_left_group) for val, func in result_funcs.iteritems()})
        else:
            #print ("Choice6", depth  )
            # Continue splitting right box. 
            self.right            = Node(self.features[~self.split_left_group], max_depth=self.max_depth, min_size=self.min_size, training_weights = self.training_weights[~self.split_left_group], training_diff_weights = self.training_diff_weights[~self.split_left_group], split_method=self.split_method, depth=self.depth+1 )

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
    def print_tree(self, key = 'FI', depth=0):
        print('%s[X%d <= %.3f]' % ((self.depth*' ', self.split_i_feature, self.split_value)))
        for node in [self.left, self.right]:
            node.print_tree(key = key, depth = depth+1)

    def total_FI(self):
        result = 0
        for node in [self.left, self.right]:
            result += node.FI if isinstance(node, ResultNode) else node.total_FI()
        return result

class ResultNode:
    ''' Simple helper class to store result value.
    '''
    def __init__( self, **kwargs ):
        for key, val in kwargs.iteritems():
            setattr( self, key, val )
    def print_tree(self, key = 'FI', depth=0):
        print('%s[%s] (%d)' % (((depth)*' ', getattr( self, key), self.size)))

if __name__=="__main__":

    import uproot
    import awkward
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
    branches    = [ "mva_photon_pt" , "mva_photon_eta", "mva_photonJetdR", "mva_photonLepdR", "mva_mT" ]
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

    #FI_func = lambda coeffs: w.get_fisherInformation_matrix( coeffs, variables = ['cWWW'], cWWW=1)[1][0][0]
    weight_mask = w.get_weight_mask( cWWW=1 )
    diff_weight_mask = w.get_diff_mask( 'cWWW', cWWW=1 )

    #TODO:
    training_weights         = np.dot(weights, w.get_weight_mask(cWWW=1))
    training_diff_weights    = np.dot(weights, w.get_diff_mask('cWWW', cWWW=1))

    print("number of events %d" % len(features))
    tic_overall = time.time()

    max_depth = 4

    for split_method in ['python_loop', 'vectorized_split_and_weight_sums']:
        tic  = time.time()
        node = Node( features, max_depth=max_depth, min_size=min_size, training_weights=training_weights, training_diff_weights=training_diff_weights, split_method=split_method )
        toc = time.time()
        print("tree construction in {time:0.4f} seconds for split_method {method:s}".format(time=toc-tic, method=split_method))
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
