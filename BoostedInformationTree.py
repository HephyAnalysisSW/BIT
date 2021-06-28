#!/usr/bin/env python
# Standard imports
import cProfile
import sys

import numpy as np

from Node import Node

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
            self.training_diff_weights+= -self.learning_rate*np.multiply(self.training_weights, np.array([root.predict(feature) for feature in self.training_features]))

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

    #self.training_diff_weights+= -self.learning_rate*np.multiply(self.training_weights, np.array([root.predict(feature) for feature in self.training_features]))

# Issues: 1. Occasional problems with accidental dtype='int' in inputs -> should be float
#         2. Deal with the pathological case where all delta-FI=0 when splitting. (We don't always want to split) 
#            --> should be done by accepting such fplits with <=float('inf'), but maybe needs check
#         3. Clean up dependencies ?
