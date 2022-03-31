#!/usr/bin/env python
# Standard imports
import cProfile
import sys
import time
import pickle

import numpy as np

import Node

default_cfg = {
    "n_trees" : 100, 
    "learning_rate" : "auto", 
    "weights_update_method" : "vectorized", 
    "calibrated" : False,
    "global_score_subtraction": False,
    "bagging_fraction": 1.,
}

class BoostedInformationTree:

    def __init__( self, training_features, training_weights, training_diff_weights, **kwargs ):


        # make cfg and node_cfg from the kwargs keys known by the Node
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        self.node_cfg = {}
        for (key, val) in kwargs.iteritems():
            if key in Node.default_cfg.keys():
                self.node_cfg[key] = val 
            elif key in default_cfg.keys():
                self.cfg[key]      = val
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )

        for (key, val) in self.cfg.iteritems():
                setattr( self, key, val )

        # Attempt to learn 98%. (1-learning_rate)^n_trees = 0.02 -> After the fit, the score is at least down to 2% 
        if self.learning_rate == "auto":
            self.learning_rate = 1-0.02**(1./self.n_trees)

        self.training_weights       = training_weights
        self.training_diff_weights  = np.copy(training_diff_weights) # Protect the outside from reweighting.
        self.training_features      = training_features

        # recall the global score
        if self.cfg["global_score_subtraction"]:
            weight_diff_sum         = self.training_diff_weights.sum()
            self.global_score       = weight_diff_sum/self.training_weights.sum()
            self.training_diff_weights -= self.training_weights*self.global_score 
            print "Subtracted a global score of %3.2f" % self.global_score 

        # Will hold the trees
        self.trees                  = []

        # information for debugging
        self.debug_data = []

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
        if hasattr( old_instance, "global_score" ):
            new_instance.global_score = old_instance.global_score
        else:
            new_instance.global_score = 0
        return new_instance  

    # Do not save debug_data
    def __getstate__(self):
        d = self.__dict__
        self_dict = {k : d[k] for k in d if k != 'debug_data'}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        pickle.dump( self, file( filename, 'w' ) )

    def boost( self, debug = True):

        toolbar_width = min(20, self.n_trees)

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        weak_learner_time = 0.0
        update_time = 0.0

        for n_tree in range(self.n_trees):

            # bagging mask
            if self.bagging_fraction<1:
                bagging_mask = np.random.choice(a=[True, False], size=len(self.training_features), p=[self.bagging_fraction, 1-self.bagging_fraction])
            else:
                bagging_mask = np.ones(len(self.training_features)).astype('bool')
            # fit to data
            time1 = time.time()
            root = Node.Node(   self.training_features[bagging_mask], 
                                training_weights        =   self.training_weights[bagging_mask], 
                                training_diff_weights   =   self.training_diff_weights[bagging_mask],

                            **self.node_cfg 
                        )
            time2 = time.time()
            weak_learner_time += time2 - time1

            # Recall current tree
            self.trees.append( root )

            # Add debug information on the first split    
            if debug:
                #print "n_tree",n_tree
                #root.print_tree()
                #print 

                self.debug_data.append( {
                    'split_i_feature': root.split_i_feature,
                    'split_value':     root.split_value,
                    'mask':            np.copy(bagging_mask),
                    #'features':      np.copy(self.training_features[bagging_mask][:, root.split_i_feature]),
                    'weights' :        np.copy(self.training_weights[bagging_mask]),
                    'diff_weights' :   np.copy(self.training_diff_weights[bagging_mask]),
                    })
 
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
       
        self.calibration_min_fac = (0., 1.)
        time1 = time.time()
        if self.calibrated:
            predictions = self.vectorized_predict(self.training_features)
            min_        = np.min(predictions)
            self.calibration_min_fac = ( min_, 1./(np.max(predictions)-min_) )
        time2 = time.time()
        calibration_time = time2 - time1
        print "calibration time: %.2f" % calibration_time

        # purge training data
        del self.training_weights       
        del self.training_diff_weights  
        del self.training_features      

    def predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False, add_global_score = True):
        # list learning rates
        learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
            learning_rates[-1] = 1
            
        predictions = np.array([ tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ])

        # Add back the global score if it was subtracted
        if self.cfg["global_score_subtraction"] and add_global_score:
            predictions += self.global_score
             
        if summed:
            return ( np.dot(learning_rates, predictions) - self.calibration_min_fac[0])*self.calibration_min_fac[1]
        else:
            return ( learning_rates*predictions - self.calibration_min_fac[0])*self.calibration_min_fac[1]
    
    def vectorized_predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False, add_global_score = True):
        # list learning rates
        learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
            learning_rates[-1] = 1
            
        predictions = np.array([ tree.vectorized_predict( feature_array ) for tree in self.trees[:max_n_tree] ])

        # Add back the global score if it was subtracted
        if self.cfg["global_score_subtraction"] and add_global_score:
            predictions += self.global_score
            
        if summed:
            return (np.dot(learning_rates, predictions) - self.calibration_min_fac[0])*self.calibration_min_fac[1]
        else:
            return (learning_rates.reshape(-1, 1)*predictions - self.calibration_min_fac[0])*self.calibration_min_fac[1]
