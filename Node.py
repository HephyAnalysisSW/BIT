#!/usr/bin/env python

import numpy as np
import operator 
from math import sqrt

class Node:
    def __init__( self, features, max_depth, min_size, training_weights, training_diff_weights, split_method="vectorized_split_and_weight_sums", _depth=0, max_relative_score_uncertainty=0):

        ## basic BDT configuration
        self.max_depth  = max_depth
        self.min_size   = min_size

        # minimum statistical uncertainty
        self.max_relative_score_uncertainty = max_relative_score_uncertainty
        assert self.max_relative_score_uncertainty==0 or split_method == "iterative_split_and_weight_sums", "max_relative_score_uncertainty argument only for split_method=iterative_split_and_weight_sums"

        # data set
        self.features   = features

        self.training_weights      = training_weights
        self.training_diff_weights = training_diff_weights       
 
        assert len(self.features) == len(self.training_weights) == len(self.training_diff_weights), "Unequal length!"

        self.size       = len(self.features)

        # keep track of recursion depth
        self._depth      = _depth
        self.split_method = split_method

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

    def jackknife_score_relative_uncertainty(self, group):
        n      = np.sum(group)
        if n<=1: return 0
        training_diff_weights_group_sum = np.sum(self.training_diff_weights[group])
        training_weights_group_sum      = np.sum(self.training_weights[group])
        all_but_one      = ( -self.training_diff_weights[group]+training_diff_weights_group_sum ) / (-self.training_weights[group]+training_weights_group_sum )
        mean_all_but_one = np.mean(all_but_one)
        uncertainty      = sqrt( (n-1.)/n*( np.sum( ( all_but_one-mean_all_but_one)**2 ) )  )
        return uncertainty/training_weights_group_sum 

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
            weight_diff_sum = 0. 
            weight_diff_sums= []
            for position, value in sorted(enumerate(feature_values), key=operator.itemgetter(1)):
                weight_sum          = weight_sum+self.training_weights[position]
                weight_sums.append( (value,  weight_sum) )
                weight_diff_sum     = weight_diff_sum+self.training_diff_weights[position]
                weight_diff_sums.append(  weight_diff_sum )

            total_weight = weight_sums[-1][1]
            total_diff_weight = weight_diff_sums[-1]
            for i_value, (value, weight_sum) in enumerate(weight_sums):
                if weight_sum==0 or total_weight==weight_sum: continue

                # only evaluate last weight sum on plateau, we cannot split indistinguishable feature values
                if i_value < len(weight_sums)-1 and weight_sums[i_value+1][0] == value:
                    continue

                # avoid splits that violate min size for one of the successors
                # TODO: we have an issue, when the allowed split range consists of one single plateau
                if i_value<self.min_size-1 or i_value>self.size-self.min_size-1: continue 

                gain = weight_diff_sums[i_value]**2/weight_sum + (total_diff_weight-weight_diff_sums[i_value])**2/(total_weight-weight_sum) 
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
            weight_sums      = np.cumsum(self.training_weights[feature_sorted_indices])
            weight_diff_sums = np.cumsum(self.training_diff_weights[feature_sorted_indices])

            # respect min size for split
            plateau_and_split_range_mask = np.ones(self.size-1, dtype=np.dtype('bool'))
            if self.min_size > 1:
                plateau_and_split_range_mask[0:self.min_size-1] = False
                plateau_and_split_range_mask[-self.min_size+1:] = False
            plateau_and_split_range_mask &= (np.diff(feature_values[feature_sorted_indices]) != 0)
            plateau_and_split_range_mask = plateau_and_split_range_mask.astype(int)

            #tic = time.time()
            self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(len(self.features), dtype='bool')
            total_weight_sum         = sorted_weight_sums[-1]
            total_diff_weight_sum    = sorted_weight_diff_sums[-1]
            sorted_weight_sums       = sorted_weight_sums[0:-1]
            sorted_weight_diff_sums  = sorted_weight_diff_sums[0:-1]
            fisher_information_left  = sorted_weight_diff_sums*sorted_weight_diff_sums/sorted_weight_sums 
            fisher_information_right = (total_diff_weight_sum-sorted_weight_diff_sums)*(total_diff_weight_sum-sorted_weight_diff_sums)/(total_weight_sum-sorted_weight_sums) 
            fisher_gains = fisher_information_left + fisher_information_right
            #print fisher_gains, fisher_information_left, fisher_information_right
            idx = np.argmax(np.nan_to_num(fisher_gains)*plateau_and_split_range_mask)
            gain =  fisher_gains[argmax_fi]
            #toc = time.time()

            #print("vectorized split in {time:0.4f} seconds".format(time=toc-tic))
            value = feature_values[feature_sorted_indices[idx]]

            if gain > self.split_gain: 
                self.split_i_feature = i_feature
                self.split_value     = value
                self.split_gain     = gain

        assert not np.isnan(self.split_value)

        #print self.split_i_feature, self.split_value, self.split_gain


    @staticmethod
    def quantile_thresholds(  n_threshold ):
        q = np.arange(0,1,1./max(n_threshold-1,1))
        q = np.append(q, [1.])
        return q

    def get_split_iterative( self ):
        ''' determine where to split the features by iteratively chopping up the sorted events
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_gain, self.split_left_group = 0, -float('inf'), 0, None

        # for a valid binary split, we need at least twice the mean size
        assert self.size >= 2*self.min_size

        # quantiles every, e.g., 5%. The "0" and "1" quantile contain the events chopped off at the "low" and "high" thresholds
        n_threshold = 4
        n_threshold = min(n_threshold, len(self.features))
        q = self.quantile_thresholds(n_threshold)
        #print "quantile thresholds:", len(q), q

        # compute the totals
        weight_sum      = np.sum(self.training_weights)
        weight_diff_sum = np.sum(self.training_diff_weights)

        for i_feature in range(len(self.features[0])):

            # inititalize
            features                = self.features[:,i_feature]
            args_sorted             = np.argsort(features)

            training_weights        = self.training_weights
            training_diff_weights   = self.training_diff_weights

            # chop off low & high end
            weight_sum_low          = np.sum(self.training_weights[args_sorted[:self.min_size]])
            weight_diff_sum_low     = np.sum(self.training_diff_weights[args_sorted[:self.min_size]])

            # special case: 2x min_size, i.e. nothing to look for
            if self.size == 2*self.min_size:
                gain = 0
                if weight_sum_low>0:
                    gain+= weight_diff_sum_low**2/weight_sum_low
                if weight_sum - weight_sum_low>0:
                    gain+= ( weight_diff_sum - weight_diff_sum_low)**2/(weight_sum-weight_sum_low)
                if gain > self.split_gain: 
                    self.split_i_feature = i_feature
                    self.split_value     = features[args_sorted[self.min_size]]
                    self.split_gain      = gain 
                continue

            # initial interval should be valid because we have at least 2x min_size events
            args_sorted             = args_sorted[self.min_size:-self.min_size] if self.min_size>0 else args_sorted
            last_round = len(args_sorted)<=len(q)
            counter    = 0

            remove_low, remove_high = 0, 0

            while True:

                remove_high = -remove_high if remove_high>0 else None 
                args_sorted = args_sorted[remove_low: remove_high]

                #print "Total:",len(digitized_features), "remove low:", remove_low, "remove_high",remove_high, "keep",np.count_nonzero(digitized_features[digitized_features==argmax_fi]),np.count_nonzero(digitized_features[digitized_features==argmax_fi+1])

                if len(args_sorted)<=n_threshold: 
                    last_round = True
                    n_threshold= len(args_sorted)
                    q = self.quantile_thresholds(n_threshold)

                quantiles          = np.quantile( features[args_sorted], q, interpolation='nearest')

                # Removing the quantiles with too high relative stat uncertainty in the score
                if self.max_relative_score_uncertainty>0:
                    #print "Removing quantiles "
                    for i_quantile, quantile in reversed(list(enumerate(quantiles))):
                        group = features<quantile
                        #print np.count_nonzero(group), np.count_nonzero(~group), group
                        #print i_quantile, quantile, np.sum(training_weights[group]), 'JN',self.jackknife_score_relative_uncertainty(group), self.jackknife_score_relative_uncertainty(~group)
                        max_jackknive_rel_uncertainty = max( map( np.nan_to_num, [self.jackknife_score_relative_uncertainty(group), self.jackknife_score_relative_uncertainty(~group)] ) )
                        if not max_jackknive_rel_uncertainty<=self.max_relative_score_uncertainty:
                            #print "Deleting i_quantile", i_quantile,  "quantile", quantile, "because max_jackknive_rel_uncertainty", max_jackknive_rel_uncertainty, np.count_nonzero(group), np.count_nonzero(~group), group, " len(quantiles) after deletion", len(quantiles)-1 
                            quantiles = np.delete(quantiles, i_quantile)
                            n_threshold-=1
                    #print "After deletion:", n_threshold, len(quantiles), quantiles
                    if n_threshold<0: break
                digitized_features = np.digitize( features[args_sorted], quantiles )

                #print
                #print "len(arg_sorted)", len(args_sorted), "last round?", last_round, "i_feature", i_feature 
                #print "features", features, "features[args_sorted]", features[args_sorted] 
                #print "quantiles", quantiles, "q",q
                #print "digitized_featues", digitized_features

                weight_sums      = weight_sum_low      + np.cumsum( [ np.sum(training_weights[args_sorted][digitized_features==i]) for i in range( n_threshold +1) ] )
                weight_diff_sums = weight_diff_sum_low + np.cumsum( [ np.sum(training_diff_weights[args_sorted][digitized_features==i]) for i in range( n_threshold  +1) ] )

                #print "weight_sum", weight_sum
                #print "weight_diff_sum", weight_diff_sum
                #print "weight_sums", weight_sums
                #print "weight_diff_sums", weight_diff_sums

                fisher_information_left  = np.divide( weight_diff_sums*weight_diff_sums, weight_sums, out=np.zeros_like(weight_diff_sums), where=weight_sums!=0) 
                weight_diff_sums_right   = weight_diff_sum-weight_diff_sums
                weight_sums_right        = weight_sum-weight_sums
                fisher_information_right = np.divide( weight_diff_sums_right*weight_diff_sums_right, weight_sums_right, out=np.zeros_like(weight_diff_sums_right), where=weight_sums_right!=0) 

                #print "fisher_information_left",  fisher_information_left
                #print "fisher_information_right", fisher_information_right

                fisher_gains = fisher_information_left + fisher_information_right
                argmax_fi    = np.argmax(np.nan_to_num(fisher_gains))
                #print "fisher_gains", fisher_gains
                #print "argmax_fi", argmax_fi, "split-value:", quantiles[argmax_fi-1]

                remove_low  =  np.count_nonzero(digitized_features[digitized_features<argmax_fi])
                remove_high =  np.count_nonzero(digitized_features[digitized_features>argmax_fi+1])

                weight_sum_low      += np.sum(training_weights[args_sorted[:remove_low]])
                weight_diff_sum_low += np.sum(training_diff_weights[args_sorted[:remove_low]])

                if last_round: 
                    #result  = quantiles[argmax_fi-1]
                    if fisher_gains[argmax_fi] > self.split_gain: 
                        self.split_i_feature = i_feature
                        self.split_value     = quantiles[argmax_fi-1]  # The implementation produces X for "feature<X", we implement "feature<=X", i.e., must have the preceding entry
                        self.split_gain      = fisher_gains[argmax_fi]
                    break

                counter += 1

            #print "Number of iterations", counter
            #print "Result (with '<=')", self.split_value

        assert not np.isnan(self.split_value)
        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(len(self.features), dtype='bool')

    # Create child splits for a node or make terminal
    def split(self, _depth=0):

        # Find the best split
        #tic = time.time()
        if self.split_method == "python_loop":
            self.get_split_python_loop()
        elif self.split_method == "vectorized_split_and_weight_sums":
            self.get_split_vectorized()
        elif self.split_method == "iterative_split_and_weight_sums":
            self.get_split_iterative()
        else:
            raise ValueError("no such split method %s" % self.split_method)

        #print("get_split in {time:0.4f} seconds".format(time=toc-tic))

        # decide what we put in the result node
        result_funcs = { 
            'size':  lambda group: np.count_nonzero(group),
            'FI'  :  lambda group: self.FI_from_group(group),
            'score': lambda group: self.score_from_group(group)
            }

        #print "left", self.jackknife_score_relative_uncertainty(self.split_left_group), "right", self.jackknife_score_relative_uncertainty(~self.split_left_group)

        # check for max depth or a 'no' split
        if  self.max_depth <= _depth+1 or (not any(self.split_left_group)) or all(self.split_left_group): # Jason Brownlee starts counting depth at 1, we start counting at 0, hence the +1
            #print ("Choice2", _depth, result_func(self.split_left_group), result_func(~self.split_left_group) )
            # The split was good, but we stop splitting further. Put everything in the left node! 
            self.split_value = float('inf')
            self.left        = ResultNode(**{val:func(np.ones(len(self.features),dtype=bool)) for val, func in result_funcs.iteritems()})
            self.right       = ResultNode(**{val:func(np.zeros(len(self.features),dtype=bool)) for val, func in result_funcs.iteritems()})
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
            self.left             = Node(self.features[self.split_left_group], max_depth=self.max_depth, min_size=self.min_size, training_weights = self.training_weights[self.split_left_group], training_diff_weights = self.training_diff_weights[self.split_left_group], split_method=self.split_method, _depth=self._depth+1 )
        # process right child
        if np.count_nonzero(~self.split_left_group) < 2*self.min_size:
            #print ("Choice5", _depth, result_func(~self.split_left_group) )
            # Too few events in the right box. We stop.
            self.right            = ResultNode(**{val:func(~self.split_left_group) for val, func in result_funcs.iteritems()})
        else:
            #print ("Choice6", _depth  )
            # Continue splitting right box. 
            self.right            = Node(self.features[~self.split_left_group], max_depth=self.max_depth, min_size=self.min_size, training_weights = self.training_weights[~self.split_left_group], training_diff_weights = self.training_diff_weights[~self.split_left_group], split_method=self.split_method, _depth=self._depth+1 )

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
