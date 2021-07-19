#!/usr/bin/env python
# Standard imports
import cProfile
import sys
import time
import pickle

import numpy as np

from NodeP3 import Node

class BoostedInformationTree:

    def __init__( self, training_features, training_weights, training_diff_weights, n_trees = 100, learning_rate = "auto", weights_update_method = "python_loop", calibrated = False, **kwargs ):

        self.n_trees        = n_trees
        self.learning_rate  = learning_rate
        self.calibrate      = calibrated 
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
        new_instance = cls( None, None, None, 
                n_trees = old_instance.n_trees, 
                learning_rate = old_instance.learning_rate, 
                weights_update_method = old_instance.weights_update_method,
                calibrated = old_instance.calibrated if hasattr(old_instance, "calibrated") else False,
                )
        new_instance.trees = old_instance.trees
        if hasattr( old_instance, "calibration_min_fac" ):
            new_instance.calibration_min_fac = old_instance.calibration_min_fac
        else:
            new_instance.calibration_min_fac = ( 0, 1 )
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
        print("weak learner time: %.2f" % weak_learner_time)
        print("update time: %.2f" % update_time)
       
        self.calibration_min_fac = (0., 1.)
        time1 = time.time()
        if self.calibrate:
            predictions = self.vectorized_predict(self.training_features)
            min_        = np.min(predictions)
            self.calibration_min_fac = ( min_, 1./(np.max(predictions)-min_) )
        time2 = time.time()
        calibration_time = time2 - time1
        print("calibration time: %.2f" % calibration_time)

        # purge training data
        del self.training_weights       
        del self.training_diff_weights  
        del self.training_features      

    def predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
        # list learning rates
        learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
            learning_rates[-1] = 1
            
        predictions = np.array([ tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ])

        if summed:
            return ( np.dot(learning_rates, predictions) - self.calibration_min_fac[0])*self.calibration_min_fac[1]
        else:
            return ( learning_rates*predictions - self.calibration_min_fac[0])*self.calibration_min_fac[1]
    
    def vectorized_predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
        # list learning rates
        learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
            learning_rates[-1] = 1
            
        predictions = np.array([ tree.vectorized_predict( feature_array ) for tree in self.trees[:max_n_tree] ])
            
        if summed:
            return (np.dot(learning_rates, predictions) - self.calibration_min_fac[0])*self.calibration_min_fac[1]
        else:
            return (learning_rates.reshape(-1, 1)*predictions - self.calibration_min_fac[0])*self.calibration_min_fac[1]
