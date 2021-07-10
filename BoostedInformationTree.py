#!/usr/bin/env python
# Standard imports
import cProfile
import sys
import time
import pickle

import numpy as np

from Node import Node

class BoostedInformationTree:

    def __init__( self, training_features, training_weights, training_diff_weights, n_trees = 100, learning_rate = "auto", weights_update_method = "python_loop", **kwargs ):

        self.n_trees        = n_trees
        self.learning_rate  = learning_rate
       
        # Attempt to learn 98%. (1-learning_rate)^n_trees = 0.02 -> After the fit, the score is at least down to 2% 
        if learning_rate == "auto":
            self.learning_rate = 1-0.02**(1./self.n_trees)
        self.kwargs         = kwargs

        self.training_weights       = training_weights
        self.training_diff_weights  = np.copy(training_diff_weights) # Protect the outside from reweighting.
        self.training_features      = training_features

        # how to update the weights in the boosting/learning process
        self.weights_update_method = weights_update_method

        # Will hold the trees
        self.trees                  = []

    @classmethod
    def load(cls, filename):
        old_instance = pickle.load(file( filename ))
        new_instance = cls( None, None, None, n_trees = old_instance.n_trees, learning_rate = old_instance.learning_rate, weights_update_method = old_instance.weights_update_method)
        new_instance.trees = old_instance.trees
        return new_instance  

    def save(self, filename):
        pickle.dump( self, file( filename, 'w' ) )

    def boost( self ):

        toolbar_width = min(20, self.n_trees)

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        weak_learner_time = 0.0
        update_time = 0.0

        for n_tree in range(self.n_trees):

            # fit to data
            time1 = time.time()
            root = Node(    self.training_features, 
                            training_weights        =   self.training_weights, 
                            training_diff_weights   =   self.training_diff_weights,
                            **self.kwargs 
                        )
            time2 = time.time()
            weak_learner_time += time2 - time1

            # Recall current tree
            self.trees.append( root )

            #print "max/min", max(training_diff_weights), min(training_diff_weights)
            #root.print_tree()
            #histo = score_histo(root)
            # Except for the last node, only take a fraction of the score

            # reduce the score
            time1 = time.time()
            if self.weights_update_method == "python_loop":
                weights_update_delta = np.multiply(self.training_weights, np.array([root.predict(feature) for feature in self.training_features]))
            elif self.weights_update_method == "vectorized":
                weights_update_delta = np.multiply(self.training_weights, root.vectorized_predict(self.training_features, key='score'))
            else:
                raise ValueError("weights update method %s unknown" % self.weights_update_method)
            self.training_diff_weights+= -self.learning_rate*weights_update_delta
            time2 = time.time()
            update_time += time2 - time1

            #np.testing.assert_array_equal(weights_update_delta, weights_update_delta_2, verbose=True)

            # update the bar
            if self.n_trees>=toolbar_width:
                if n_tree % (self.n_trees/toolbar_width)==0:   sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("]\n") # this ends the progress bar
        print "weak learner time: %.2f" % weak_learner_time
        print "update time: %.2f" % update_time

        # purge training data
        self.training_weights       
        self.training_diff_weights  
        self.training_features      

    def predict( self, feature_array, max_n_tree = None, summed = True, vectorized = False):
        if vectorized:
            predictions = [ self.learning_rate*tree.vectorized_predict( feature_array ) for tree in self.trees[:max_n_tree] ]
        else:    
            predictions = [ self.learning_rate*tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ]

        if summed:
            if vectorized:
                return np.sum(predictions, axis=0)
            else: 
                return sum( predictions, 0. )
        else:
            return np.array(predictions)
