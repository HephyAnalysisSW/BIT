#!/usr/bin/env python

import numpy as np
import operator 
from math import sqrt

default_cfg = {
    "max_depth": 2,
    "min_size" : 50,
    "max_uncertainty": -1,
    "split_method":"vectorized_split_and_weight_sums",
    "max_n_split": -1,
}

class Node:
    def __init__( self, features, training_weights, training_diff_weights, _depth=0, **kwargs):

        ## basic BDT configuration + kwargs
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        for attr, val in self.cfg.iteritems():
            setattr( self, attr, val )

        self.epsilon                      = 1e-10

        # data set
        self.features   = features

        self.training_weights      = training_weights.astype('float')
        self.training_diff_weights = training_diff_weights.astype('float')
 
        self.size        = len(self.features)
        assert self.size == len(self.training_weights) == len(self.training_diff_weights), "Unequal length!"

        # keep track of recursion depth
        self._depth      = _depth

        self.split(_depth=_depth)
        self.prune()

        # Let's not leak the dataset.
        del self.training_weights
        del self.training_diff_weights
        del self.features 
        del self.split_left_group 

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

    def get_split_python_loop( self ):
        ''' determine where to split the features (python loop)
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_gain, self.split_left_group = 0, -float('inf'), 0, None

        # for a valid binary split, we need at least twice the mean size
        assert self.size >= 2*self.min_size

        # loop over features
        for i_feature in range(len(self.features[0])):
            feature_values = self.features[:,i_feature]

            weight_sum = 0. 
            weight_sums= []
            diff_weight_sum = 0. 
            diff_weight_sums= []
            for position, value in sorted(enumerate(feature_values), key=operator.itemgetter(1)):
                weight_sum          = weight_sum+self.training_weights[position]
                weight_sums.append( (value,  weight_sum) )
                diff_weight_sum     = diff_weight_sum+self.training_diff_weights[position]
                diff_weight_sums.append(  diff_weight_sum )

            total_weight = weight_sums[-1][1]
            total_diff_weight = diff_weight_sums[-1]
            for i_value, (value, weight_sum) in enumerate(weight_sums):
                if weight_sum==0 or total_weight==weight_sum: continue

                # only evaluate last weight sum on plateau, we cannot split indistinguishable feature values
                if i_value < len(weight_sums)-1 and weight_sums[i_value+1][0] == value:
                    continue

                # avoid splits that violate min size for one of the successors
                # TODO: we have an issue, when the allowed split range consists of one single plateau
                if i_value<self.min_size-1 or i_value>self.size-self.min_size-1: continue 

                gain = diff_weight_sums[i_value]**2/weight_sum + (total_diff_weight-diff_weight_sums[i_value])**2/(total_weight-weight_sum) 
                if gain > self.split_gain: 
                    self.split_i_feature = i_feature
                    self.split_value     = value
                    self.split_gain      = gain

        assert not np.isnan(self.split_value)
        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value
        #print "python_loop", self.split_i_feature, self.split_value,  self.split_left_group 

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

            # stable sort uses timsort like sorted in python
            #feature_sorted_indices = np.argsort(feature_values, kind='stable')
            feature_sorted_indices = np.argsort(feature_values)
            sorted_weight_sums     = np.cumsum(self.training_weights[feature_sorted_indices])
            sorted_diff_weight_sums= np.cumsum(self.training_diff_weights[feature_sorted_indices])
            
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
            plateau_and_split_range_mask = plateau_and_split_range_mask.astype(int)

            if self.max_uncertainty>0:
                n           = len(feature_values)
                n_arr_left  = np.arange(1,n+1).astype('float')
                n_arr_right = n-n_arr_left

                sorted_weight_squared_sums      = np.cumsum( (self.training_weights[feature_sorted_indices]      - np.divide(sorted_weight_sums,n_arr_left,      where=n_arr_left>0,out=np.zeros_like(n_arr_left))) **2)
                sorted_diff_weight_squared_sums = np.cumsum( (self.training_diff_weights[feature_sorted_indices] - np.divide(sorted_diff_weight_sums,n_arr_left, where=n_arr_left>0,out=np.zeros_like(n_arr_left))) **2)

                total_weight_squared_sums  = sorted_weight_squared_sums[-1]
                sorted_weight_squared_sums = sorted_weight_squared_sums[0:-1]

                total_diff_weight_squared_sums  = sorted_diff_weight_squared_sums[-1]
                sorted_diff_weight_squared_sums = sorted_diff_weight_squared_sums[0:-1]

                n_arr_left  = n_arr_left[0:-1]
                n_arr_right = n_arr_right[0:-1]
                n_fac_left  = np.divide(n_arr_left,  n_arr_left-1,  where=n_arr_left>1, out=np.zeros_like(n_arr_left-1))
                n_fac_right = np.divide(n_arr_right, n_arr_right-1, where=n_arr_right>1,out=np.zeros_like(n_arr_right-1))

                variance_lambda_left  = n_fac_left*sorted_weight_squared_sums
                variance_lambda_right = n_fac_right*(total_weight_squared_sums-sorted_weight_squared_sums)
                variance_diff_lambda_left  = n_fac_left*sorted_diff_weight_squared_sums
                variance_diff_lambda_right = n_fac_right*(total_diff_weight_squared_sums-sorted_diff_weight_squared_sums)

                #print "n_arr_left", n_arr_left
                #print "np.divide(n_arr_left, n_arr_left-1, where=n_arr_left>1)", np.divide(n_arr_left, n_arr_left-1, where=n_arr_left>1)
                #print "self.training_weights[feature_sorted_indices]", self.training_weights[feature_sorted_indices[:-1]][:100]
                #print "sorted_weight_sums", sorted_weight_sums[:100] 
                #print "np.divide(sorted_weight_sums,n_arr_left, where=n_arr_left>1)",  np.divide(sorted_weight_sums,n_arr_left, where=n_arr_left>1)[:100]
                #print "res", (np.divide(n_arr_left, n_arr_left-1, where=n_arr_left>1) * np.cumsum( (self.training_weights[feature_sorted_indices[:-1]] - np.divide(sorted_weight_sums,n_arr_left, where=n_arr_left>1)) **2 ))[:100]
                #print 

            #tic = time.time()
            total_weight_sum         = sorted_weight_sums[-1]
            total_diff_weight_sum    = sorted_diff_weight_sums[-1]
            sorted_weight_sums       = sorted_weight_sums[0:-1]
            sorted_diff_weight_sums  = sorted_diff_weight_sums[0:-1]
            fisher_information_left  = sorted_diff_weight_sums**2/sorted_weight_sums

            sorted_diff_weight_sums_right = total_diff_weight_sum-sorted_diff_weight_sums
            sorted_weight_sums_right = total_weight_sum-sorted_weight_sums
            fisher_information_right = sorted_diff_weight_sums_right**2/sorted_weight_sums_right

            fisher_gains = fisher_information_left + fisher_information_right
            
            if self.max_uncertainty>0:
                #relative_information_variance_left  = 4.*variance_lambda_left/sorted_weight_sums**2 + variance_diff_lambda_left/sorted_diff_weight_sums**2
                #relative_information_variance_right = 4.*variance_lambda_right/(total_weight_sum-sorted_weight_sums)**2 + variance_diff_lambda_right/(total_diff_weight_sum-sorted_diff_weight_sums)**2

                den_left = sorted_weight_sums*sorted_diff_weight_sums_right - sorted_diff_weight_sums*sorted_weight_sums_right
                den_right= sorted_weight_sums_right*sorted_diff_weight_sums - sorted_diff_weight_sums_right*sorted_weight_sums
                relative_information_variance = variance_lambda_left*  (sorted_weight_sums_right/sorted_weight_sums)**2* (1./total_weight_sum + 2*sorted_diff_weight_sums/den_left )**2\
                                              + variance_lambda_right* (sorted_weight_sums/sorted_weight_sums_right)**2* (1./total_weight_sum + 2*sorted_diff_weight_sums_right/den_right )**2\
                                              + variance_diff_lambda_left*(2.*sorted_weight_sums_right/den_left)**2\
                                              + variance_diff_lambda_right*(2.*sorted_weight_sums/den_right)**2
                uncertainty_mask = ( (relative_information_variance>0) & (relative_information_variance<(self.max_uncertainty**2))).astype('int')

                argmax_fi = np.argmax(np.nan_to_num(fisher_gains)*plateau_and_split_range_mask*uncertainty_mask)
            else:
                argmax_fi = np.argmax(np.nan_to_num(fisher_gains)*plateau_and_split_range_mask)

            gain      =  fisher_gains[argmax_fi]
            #toc = time.time()

            #print("vectorized split in {time:0.4f} seconds".format(time=toc-tic))
            value = feature_values[feature_sorted_indices[argmax_fi]]

            if gain > self.split_gain: 
                self.split_i_feature = i_feature
                self.split_value     = value
                self.split_gain      = gain

        assert not np.isnan(self.split_value)

        #print self.split_i_feature, self.split_value, self.split_gain
        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(self.size, dtype='bool')

    @staticmethod
    def quantile_thresholds(  n_threshold ):
        q = np.arange(0,1,1./max(n_threshold-1,1))
        q = np.append(q, [1.])
        return q

    # Create child splits for a node or make terminal
    def split(self, _depth=0):

        # Find the best split
        #tic = time.time()
        if self.split_method == "python_loop":
            self.get_split_python_loop()
        elif self.split_method == "vectorized_split_and_weight_sums":
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
        if  self.max_depth <= _depth+1 or (not any(self.split_left_group)) or all(self.split_left_group): # Jason Brownlee starts counting depth at 1, we start counting at 0, hence the +1
            #print ("Choice2", _depth, result_func(self.split_left_group), result_func(~self.split_left_group) )
            # The split was good, but we stop splitting further. Put everything in the left node! 
            self.split_value = float('inf')
            self.left        = ResultNode(**{val:func(np.ones(self.size,dtype=bool)) for val, func in result_funcs.iteritems()})
            self.right       = ResultNode(**{val:func(np.zeros(self.size,dtype=bool)) for val, func in result_funcs.iteritems()})
            # The split was good, but we stop splitting further. Put the result of the split in the left/right boxes.
            #self.left, self.right = ResultNode(**{val:func(self.split_left_group) for val, func in result_funcs.iteritems()}), ResultNode(**{val:func(~self.split_left_group) for val, func in result_funcs.iteritems()})
            return
        # process left child
        if np.count_nonzero(self.split_left_group) < 2*self.min_size:
            #print ("Choice3", _depth, result_func(self.split_left_group) )
            # Too few events in the left box. We stop.
            self.left             = ResultNode(**{val:func(self.split_left_group) for val, func in result_funcs.iteritems()})
        else:
            #print ("Choice4", _depth )
            # Continue splitting left box.
            self.left             = Node(self.features[self.split_left_group], training_weights = self.training_weights[self.split_left_group], training_diff_weights = self.training_diff_weights[self.split_left_group], _depth=self._depth+1, **self.cfg)
        # process right child
        if np.count_nonzero(~self.split_left_group) < 2*self.min_size:
            #print ("Choice5", _depth, result_func(~self.split_left_group) )
            # Too few events in the right box. We stop.
            self.right            = ResultNode(**{val:func(~self.split_left_group) for val, func in result_funcs.iteritems()})
        else:
            #print ("Choice6", _depth  )
            # Continue splitting right box. 
            self.right            = Node(self.features[~self.split_left_group], training_weights = self.training_weights[~self.split_left_group], training_diff_weights = self.training_diff_weights[~self.split_left_group], _depth=self._depth+1, **self.cfg)

    # Prediction    
    #@memoized -> The decorator doesn't work because numpy.ndarrays are not hashable. Hashes are for fast lookup. Maybe it's dangerous to circumvent this with hash(ndarray.tostring())? Niki?
    def predict( self, features, key = 'score'):
        ''' obtain the result by recursively descending down the tree
        '''
        node = self.left if features[self.split_i_feature]<=self.split_value else self.right
        if isinstance(node, ResultNode):
            return getattr(node, key)
        else:
            return node.predict(features, key=key)

    def vectorized_predict(self, feature_matrix, key='score'):
        """Create numpy logical expressions from all paths to results nodes, associate with prediction defined by key, and return predictions for given feature matrix
           Should be faster for shallow trees due to numpy being implemented in C, despite going over feature vectors multiple times."""
        
        emmitted_expressions_with_predictions = []

        def emit_expressions_with_predictions(node, logical_expression):
            if isinstance(node, ResultNode):
                emmitted_expressions_with_predictions.append((logical_expression, getattr(node, key)))
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
        predictions = np.zeros(len(feature_matrix))

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
    def print_tree(self, key = 'FI', _depth=0):
        print('%s[X%d <= %.3f]' % ((self._depth*' ', self.split_i_feature, self.split_value)))
        for node in [self.left, self.right]:
            node.print_tree(key = key, _depth = _depth+1)

    def total_FI(self):
        result = 0
        for node in [self.left, self.right]:
            result += node.FI if isinstance(node, ResultNode) else node.total_FI()
        return result

    def get_list(self, key = 'score'):
        ''' recursively obtain all thresholds '''
        return [ (self.split_i_feature, self.split_value), self.left.get_list(key), self.right.get_list(key) ] 

class ResultNode:
    ''' Simple helper class to store result value.
    '''
    def __init__( self, **kwargs ):
        for key, val in kwargs.iteritems():
            setattr( self, key, val )
    def print_tree(self, key = 'FI', _depth=0):
        print('%s[%s] (%d)' % (((_depth)*' ', getattr( self, key), self.size)))

    def get_list( self, key='score'):
        ''' recursively obtain all thresholds (bottom of recursion)'''
        return getattr(self, key) 
