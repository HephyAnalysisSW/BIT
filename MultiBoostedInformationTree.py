#!/usr/bin/env python
# Standard imports
import cProfile
import sys
#sys.path.insert( 0, '..')
#sys.path.insert( 0, '.')
import time
import pickle
import copy
import itertools
import numpy as np
import operator
import functools

import MultiNode

default_cfg = {
    "n_trees" : 100,
    "learning_rate" : 0.2, 
    "loss" : "MSE", # or "CrossEntropy" 
#    "bagging_fraction": 1.,
}

class MultiBoostedInformationTree:

    def __init__( self, training_features, training_weights, **kwargs ):

        # make cfg and node_cfg from the kwargs keys known by the Node
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        self.node_cfg = {}
        for (key, val) in kwargs.items():
            if key in MultiNode.default_cfg.keys():
                self.node_cfg[key] = val 
            elif key in default_cfg.keys():
                self.cfg[key]      = val
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )
        self.node_cfg['loss'] = self.cfg['loss'] 

        for (key, val) in self.cfg.items():
                setattr( self, key, val )

        # Attempt to learn 98%. (1-learning_rate)^n_trees = 0.02 -> After the fit, the score is at least down to 2% 
        if self.learning_rate == "auto":
            self.learning_rate = 1-0.02**(1./self.n_trees)

        self.training_weights   = copy.deepcopy(training_weights)
        if training_weights is not None:
            self.training_weights   = {tuple(sorted(key)):val for key,val in self.training_weights.items()}
        self.training_features  = training_features

        # Will hold the trees
        self.trees              = []

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as file_:
            old_instance = pickle.load(file_)
            new_instance = cls( None, None, 
                    n_trees = old_instance.n_trees, 
                    learning_rate = old_instance.learning_rate, 
                    )
            new_instance.trees = old_instance.trees

            new_instance.derivatives = old_instance.trees[0].derivatives[1:]

            return new_instance  

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        with open(filename,'wb') as file_:
            pickle.dump( self, file_ )

    def boost( self ):

        toolbar_width = min(20, self.n_trees)

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        weak_learner_time = 0.0
        update_time = 0.0
        for n_tree in range(self.n_trees):

            training_time = 0

            # fit to data
            time1 = time.process_time()
            root = MultiNode.MultiNode(   
                            self.training_features, 
                            training_weights = self.training_weights,
                            **self.node_cfg 
                        )

            if n_tree==0:
                self.derivatives = root.derivatives[1:]

            time2 = time.process_time()
            weak_learner_time += time2 - time1
            training_time      = time2 - time1

            self.trees.append( root )

            # Recall current tree
            time1 = time.process_time()

            prediction   = root.vectorized_predict(self.training_features)
            len_         = len(prediction)
            delta_weight = self.training_weights[tuple()].reshape(len_,-1)*prediction[:,1:]/prediction[:,0].reshape(len_,-1)
            for i_der, der in enumerate(root.derivatives[1:]):
                self.training_weights[der] += -self.learning_rate*delta_weight[:,i_der]

            time2 = time.process_time()
            update_time   += time2 - time1
            training_time += time2 - time1

            self.trees[-1].training_time = training_time 

            # update the bar
            if self.n_trees>=toolbar_width:
                if n_tree % (self.n_trees/toolbar_width)==0:   sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("]\n") # this ends the progress bar
        print ("weak learner time: %.2f" % weak_learner_time)
        print ("update time: %.2f" % update_time)
       
        # purge training data
        del self.training_weights       
        del self.training_features      

    def predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
        # list learning rates
        learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
            learning_rates[-1] = 1
            
        predictions = np.array([ tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ])
        predictions = predictions[:,1:]/predictions[:,0].reshape(-1,1)
        if summed:
            return np.dot(learning_rates, predictions)
        else:
            return learning_rates.reshape(-1, 1)*predictions
    
    def vectorized_predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
        # list learning rates
        learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
            learning_rates[-1] = 1
            
        predictions = np.array([ tree.vectorized_predict( feature_array ) for tree in self.trees[:max_n_tree] ])
        predictions = predictions[:,:,1:]/np.expand_dims(predictions[:,:,0], -1)
        if summed:
            return np.sum(learning_rates.reshape(-1,1,1)*predictions, axis=0)
        else:
            return learning_rates.reshape(-1,1,1)*predictions 

    def losses( self, feature_array, weight_dict, max_n_tree = None, last_tree_counts_full = False):
        # list learning rates
        learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
            learning_rates[-1] = 1

        # recover base points from tree
        base_points      = self.trees[0].base_points
        base_point_const = np.array([[ functools.reduce(operator.mul, [point[coeff] if (coeff in point) else 0 for coeff in der ], 1) for der in self.derivatives] for point in base_points]).astype('float')
        for i_der, der in enumerate(self.derivatives):
            if not (len(der)==2 and der[0]==der[1]): continue
            for i_point in range(len(base_points)):
                base_point_const[i_point][i_der]/=2.
        
        predictions = np.array([ tree.vectorized_predict( feature_array ) for tree in self.trees[:max_n_tree] ])
        predictions = predictions[:,:,1:]/np.expand_dims(predictions[:,:,0], -1)

        weight_ratio = np.array( [ (weight_dict[der]/weight_dict[()] if der in weight_dict else weight_dict[tuple(reversed(der))]/weight_dict[()]) for der in self.derivatives]).transpose().astype('float')
        # losses
        return -( weight_dict[()][np.newaxis,...,np.newaxis]*np.dot( (predictions - (weight_ratio[np.newaxis,...])), base_point_const )**2).sum(axis=(1,2))
        
#max_n_tree      = None
#derivatives     = bit.derivatives
## recover base points from tree
#base_points      = bit.trees[0].base_points
#base_point_const = np.array([[ functools.reduce(operator.mul, [point[coeff] if (coeff in point) else 0 for coeff in der ], 1) for der in bit.derivatives] for point in base_points]).astype('float')
#for i_der, der in enumerate(bit.derivatives):
#    if not (len(der)==2 and der[0]==der[1]): continue
#    for i_point in range(len(base_points)):
#        base_point_const[i_point][i_der]/=2.
#
#learning_rates  = bit.learning_rate*np.ones(max_n_tree if max_n_tree is not None else bit.n_trees)
#predictions     = np.array([ tree.vectorized_predict( training_features ) for tree in bit.trees[:max_n_tree] ])
#predictions     = predictions[:,:,1:]/np.expand_dims(predictions[:,:,0], -1)
#
#weight_ratio = np.array([training_weights[der]/training_weights[()] for der in derivatives]).transpose().astype('float')
#
#losses = -( training_weights[()][np.newaxis,...,np.newaxis]*(np.dot( (predictions - (weight_ratio[np.newaxis,...])), base_point_const ))**2).sum(axis=(1,2))

if __name__=='__main__':

    #import toy_models.ZH_Nakamura as model
    #coefficients = sorted(['cHW', 'cHWtil'])

    import toy_models.analytic as model
    coefficients = sorted(['theta1'])

    base_points = []
    for comb in list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2)):
        base_points.append( {c:comb.count(c) for c in coefficients} )

    nTraining    = 50000

    features          = model.getEvents(nTraining)
    training_weights  = model.getWeights(features, eft=model.default_eft_parameters)
    print ("Created training data set of size %i" % len(features) )

    for key in training_weights.keys():
        if key==tuple(): continue
        if not all( [ k in coefficients for k in key] ):
            del training_weights[key]

    print ("nEvents: %i Weights: %s" %( len(features), [ k for k in training_weights.keys() if k!=tuple()] ))

    # cfg & preparation for node split
    min_size    = 50
    max_n_split = -1

    bit = MultiBoostedInformationTree( 
                      features,
                      training_weights,
                      min_size    = min_size,
                      max_n_split = max_n_split,
                      base_points = base_points,
                      feature_names = model.feature_names,
                    )
    bit.boost()
