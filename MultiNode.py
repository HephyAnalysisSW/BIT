#!/usr/bin/env python

import numpy as np
import operator 
from math import sqrt
import itertools

default_cfg = {
    "max_depth":        4,
    "min_size" :        50,
    "max_n_split":      -1,
    "positive_score":   False,
    "base_points":      None,
    "feature_names":    None,
    "positive":         False,
}

class MultiNode:
    def __init__( self, features, training_weights, _depth=0, **kwargs):

        ## basic BDT configuration + kwargs
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        for attr, val in self.cfg.iteritems():
            setattr( self, attr, val )

        self.epsilon                      = 1e-10

        # data set
        self.features           = features
        self.size               = len(self.features)

        # Master node: Split format
        if type(training_weights)==dict:
            self.coefficients            = sorted(list(set(sum(map(list,training_weights.keys()),[]))))

            self.first_derivatives  = sorted(list(itertools.combinations_with_replacement(self.coefficients,1))) 
            self.second_derivatives = sorted(list(itertools.combinations_with_replacement(self.coefficients,2))) 
            self.derivatives        = [tuple()] + self.first_derivatives + self.second_derivatives

            self.training_weights   = {tuple(sorted(key)):val for key,val in training_weights.iteritems()}

            assert kwargs.has_key('base_points') and kwargs['base_points'] is not None, "Must provide base_points in cfg"

            # precoumputed base_point_const
            self.base_points      = kwargs['base_points']
            self.base_point_const = np.array([[ reduce(operator.mul, [point[coeff] if point.has_key(coeff) else 0 for coeff in der ], 1) for der in self.derivatives] for point in self.base_points]).astype('float')
            for i_der, der in enumerate(self.derivatives):
                if not (len(der)==2 and der[0]==der[1]): continue
                for i_point in range(len(self.base_points)):
                    self.base_point_const[i_point][i_der]/=2.

            assert np.linalg.matrix_rank(self.base_point_const) == self.base_point_const.shape[0], \
                   "Base points not linearly independent! Found rank %i for %i base_points" %( np.linalg.matrix_rank(self.base_point_const), self.base_point_const.shape[0])

            # make another version of base_point_const that contains the [1,0,0,...] vector -> used for testing positivity of the zeroth coefficient
            const = np.zeros((1,len(self.derivatives)))
            const[0,0]=1
            self.base_point_const_for_pos = np.concatenate((const, self.base_point_const))

            self.cfg['base_point_const']         = self.base_point_const
            self.cfg['base_point_const_for_pos'] = self.base_point_const_for_pos
            self.cfg['derivatives'] = self.derivatives 
            self.cfg['feature_names'] = None if not kwargs.has_key('feature_names') else kwargs['feature_names'] 
            self.feature_names      = self.cfg['feature_names']
            self.training_weights   = np.array([training_weights[der] for der in self.derivatives]).transpose().astype('float')
        # inside tree
        else:
            self.training_weights           = training_weights
            self.base_point_const           = kwargs['base_point_const']
            self.base_point_const_for_pos   = kwargs['base_point_const_for_pos']
            self.derivatives                = kwargs['derivatives']
            self.feature_names              = kwargs['feature_names']

        # keep track of recursion depth
        self._depth             = _depth

        self.split(_depth=_depth)
        self.prune()

        # Let's not leak the dataset.
        del self.training_weights
        del self.features 
        del self.split_left_group 

    def get_split_vectorized( self ):
        ''' determine where to split the features, first vectorized version of FI maximization
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_gain, self.split_left_group = 0, -float('inf'), 0, None

        # for a valid binary split, we need at least twice the mean size
        assert self.size >= 2*self.min_size

        # loop over features
        #print "len(self.features[0]))",len(self.features[0])

        for i_feature in range(len(self.features[0])):
            feature_values = self.features[:,i_feature]

            feature_sorted_indices = np.argsort(feature_values)
            sorted_weight_sums     = np.cumsum(self.training_weights[feature_sorted_indices],axis=0) # FIXME cumsum does not respect max_n_split
 
            # respect min size for split
            if self.max_n_split<2:
                plateau_and_split_range_mask = np.ones(self.size-1, dtype=np.dtype('bool'))
            else:
                min_, max_ = min(feature_values), max(feature_values) 
                #print "_depth",self._depth, "len(feature_values)",len(feature_values), "min_, max_", min_, max_
                plateau_and_split_range_mask  = np.digitize(feature_values[feature_sorted_indices], np.arange (min_, max_, (max_-min_)/(self.max_n_split+1)))
                #print len(plateau_and_split_range_mask), plateau_and_split_range_mask
                plateau_and_split_range_mask = plateau_and_split_range_mask[1:]-plateau_and_split_range_mask[:-1]
                plateau_and_split_range_mask = np.insert( plateau_and_split_range_mask, 0, 0).astype('bool')[:-1]
                #print "plateau_and_split_range_mask", plateau_and_split_range_mask
                #print "CUTS", feature_values[feature_sorted_indices][:-1][plateau_and_split_range_mask] 

            if self.min_size > 1:
                plateau_and_split_range_mask[0:self.min_size-1] = False
                plateau_and_split_range_mask[-self.min_size+1:] = False
            plateau_and_split_range_mask &= (np.diff(feature_values[feature_sorted_indices]) != 0)

            total_weight_sum         = sorted_weight_sums[-1]
            sorted_weight_sums       = sorted_weight_sums[0:-1]
            sorted_weight_sums_right = total_weight_sum-sorted_weight_sums

            # mask negative definite splits
            if self.cfg['positive']:
                pos       = np.apply_along_axis(all, 1, np.dot(sorted_weight_sums,self.base_point_const_for_pos.transpose())>=0)
                pos_right = np.apply_along_axis(all, 1, np.dot(sorted_weight_sums_right,self.base_point_const_for_pos.transpose())>=0)

                all_pos = np.concatenate((pos, pos_right))
                #if not np.all(all_pos):
                #    print ("Warning! Found negative node splits {:.2%}".format(1-float(np.count_nonzero(all_pos))/len(all_pos)) )

                plateau_and_split_range_mask &= pos
                plateau_and_split_range_mask &= pos_right

            plateau_and_split_range_mask = plateau_and_split_range_mask.astype(int)

            neg_loss_gains = np.sum(np.dot( sorted_weight_sums, self.base_point_const.transpose())**2,axis=1)/sorted_weight_sums[:,0]
            neg_loss_gains+= np.sum(np.dot( sorted_weight_sums_right, self.base_point_const.transpose())**2,axis=1)/sorted_weight_sums_right[:,0]

            argmax_fi = np.argmax(np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask)
            gain      =  neg_loss_gains[argmax_fi]

            value = feature_values[feature_sorted_indices[argmax_fi]]

            if gain > self.split_gain: 
                self.split_i_feature = i_feature
                self.split_value     = value
                self.split_gain      = gain

        assert not np.isnan(self.split_value)

        #print self.split_i_feature, self.split_value, self.split_gain
        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(self.size, dtype='bool')

    def predict_coefficients( self, group ):
        return np.sum(self.training_weights[group],axis=0)

    # Create child splits for a node or make terminal
    def split(self, _depth=0):

        # Find the best split
        #tic = time.time()
        self.get_split_vectorized()

        # check for max depth or a 'no' split
        if  self.max_depth <= _depth+1 or (not any(self.split_left_group)) or all(self.split_left_group): # Jason Brownlee starts counting depth at 1, we start counting at 0, hence the +1
            #print ("Choice2", _depth, result_func(self.split_left_group), result_func(~self.split_left_group) )
            # The split was good, but we stop splitting further. Put everything in the left node! 
            self.split_value = float('inf')
            self.left        = ResultNode(self.predict_coefficients(np.ones(self.size,dtype=bool)),derivatives=self.derivatives)
            self.right       = ResultNode(self.predict_coefficients(np.zeros(self.size,dtype=bool)),derivatives=self.derivatives)
            # The split was good, but we stop splitting further. Put the result of the split in the left/right boxes.
            #self.left, self.right = ResultNode(**{val:func(self.split_left_group) for val, func in result_funcs.iteritems()}), ResultNode(**{val:func(~self.split_left_group) for val, func in result_funcs.iteritems()})
            return
        # process left child
        if np.count_nonzero(self.split_left_group) < 2*self.min_size:
            #print ("Choice3", _depth, result_func(self.split_left_group) )
            # Too few events in the left box. We stop.
            self.left             = ResultNode(self.predict_coefficients(self.split_left_group),derivatives=self.derivatives)
        else:
            #print ("Choice4", _depth )
            # Continue splitting left box.
            self.left             = MultiNode(self.features[self.split_left_group], training_weights = self.training_weights[self.split_left_group], _depth=self._depth+1, **self.cfg)
        # process right child
        if np.count_nonzero(~self.split_left_group) < 2*self.min_size:
            #print ("Choice5", _depth, result_func(~self.split_left_group) )
            # Too few events in the right box. We stop.
            self.right            = ResultNode(self.predict_coefficients(~self.split_left_group),derivatives=self.derivatives)
        else:
            #print ("Choice6", _depth  )
            # Continue splitting right box. 
            self.right            = MultiNode(self.features[~self.split_left_group], training_weights = self.training_weights[~self.split_left_group], _depth=self._depth+1, **self.cfg)

    # Prediction    
    def predict( self, features):
        ''' obtain the result by recursively descending down the tree
        '''
        node = self.left if features[self.split_i_feature]<=self.split_value else self.right
        if isinstance(node, ResultNode):
            return node.predicted_coefficients 
        else:
            return node.predict(features)

    def vectorized_predict(self, feature_matrix):
        """Create numpy logical expressions from all paths to results nodes, associate with prediction defined by key, and return predictions for given feature matrix
           Should be faster for shallow trees due to numpy being implemented in C, despite going over feature vectors multiple times."""

        emmitted_expressions_with_predictions = []

        def emit_expressions_with_predictions(node, logical_expression):
            if isinstance(node, ResultNode):
                emmitted_expressions_with_predictions.append((logical_expression, node.predicted_coefficients))
            else:
                if node == self:
                    prepend = ""
                else:
                    prepend = " & "
                if np.isinf(node.split_value):
                    split_value_str = 'np.inf'
                else:
                    split_value_str = format(node.split_value, '.32f')
                emit_expressions_with_predictions(node.left, logical_expression + "%s(feature_matrix[:,%d] <= %s)" % (prepend, node.split_i_feature, split_value_str))
                emit_expressions_with_predictions(node.right, logical_expression + "%s(feature_matrix[:,%d] > %s)" % (prepend, node.split_i_feature, split_value_str))

        emit_expressions_with_predictions(self, "")
        predictions = np.zeros((len(feature_matrix), len(self.derivatives)))

        for expression, prediction in emmitted_expressions_with_predictions:
            predictions[eval(expression)] = prediction

        return predictions

    # remove the 'inf' splits
    def prune( self ):
        if not isinstance(self.left, ResultNode) and self.left.split_value==float('+inf'):
            self.left = self.left.left
        elif not isinstance(self.left, ResultNode):
            self.left.prune()
        if not isinstance(self.right, ResultNode) and self.right.split_value==float('+inf'):
            self.right = self.right.left
        elif not isinstance(self.right, ResultNode):
            self.right.prune()

    # Print a decision tree
    def print_tree(self, _depth=0):
        print('%s[%s <= %.3f]' % ((self._depth*' ', "X%d"%self.split_i_feature if self.feature_names is None else self.feature_names[self.split_i_feature], self.split_value)))
        for node in [self.left, self.right]:
            node.print_tree(_depth = _depth+1)

    def get_list(self):
        ''' recursively obtain all thresholds '''
        return [ (self.split_i_feature, self.split_value), self.left.get_list(), self.right.get_list() ] 

class ResultNode:
    ''' Simple helper class to store result value.
    '''
    def __init__( self, predicted_coefficients, derivatives=None):
        self.predicted_coefficients = predicted_coefficients
        self.derivatives            = derivatives

    @staticmethod
    def prefac(der):
        return (0.5 if (len(der)==2 and len(set(der))==1) else 1. )

    def print_tree(self, _depth=0):
        #poly_str = "".join(["*".join(["{:+.3e}".format(self.predicted_coefficients[i_der])] + list(self.derivatives[i_der]) ) for i_der in range(len(self.derivatives))])
        poly_str = "".join(["*".join(["{:+.3e}".format(self.prefac(der)*self.predicted_coefficients[i_der]/self.predicted_coefficients[0])] + list(self.derivatives[i_der]) ) for i_der, der in enumerate(self.derivatives)])
        print('%s r = %s' % ((_depth)*' ', poly_str) )

    def get_list(self):
        ''' recursively obtain all thresholds (bottom of recursion)'''
        return self.predicted_coefficients 

if __name__=='__main__':

    import VH_models

    #model = VH_models.ZH_Nakamura_debug
    #coefficients = sorted(['cHW', 'cHWtil', 'cHQ3'])
    #nTraining    = 50000

    model = VH_models.analytic
    coefficients = sorted(['theta1'])
    nTraining    = 50000

    features          = model.getEvents(nTraining)
    training_weights  = model.getWeights(features, eft=model.default_eft_parameters)
    print ("Created training data set of size %i" % len(features) )

    for key in training_weights.keys():
        if key==tuple(): continue
        if not all( [ k in coefficients for k in key] ):
            del training_weights[key]

    print "nEvents: %i Weights: %s" %( len(features), [ k for k in training_weights.keys() if k!=tuple()] )

    base_points = []
    for comb in list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2)):
        base_points.append( {c:comb.count(c) for c in coefficients} )

    # cfg & preparation for node split
    min_size    = 50
    max_n_split = -1
 
    node = MultiNode( features, 
                      training_weights,
                      min_size    = min_size,
                      max_n_split = max_n_split, 
                      base_points = base_points,
                      feature_names = model.feature_names,
                    )
