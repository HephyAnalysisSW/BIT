#!/usr/bin/env python

import unittest
import numpy as np
import time
from BoostedInformationTree import BoostedInformationTree

class TestBIT(unittest.TestCase):
        def test_power_law_model(self):
                data_dir = 'data'

                training_features = np.loadtxt('%s/training_features_power_law_model.txt.gz' % data_dir)
                training_features = training_features.reshape(training_features.shape[0], -1)
                print(training_features.shape)
                training_weights = np.loadtxt('%s/training_weights_power_law_model.txt.gz' % data_dir)
                training_diff_weights = np.loadtxt('%s/training_diff_weights_power_law_model.txt.gz' % data_dir)

                learning_rate = 0.02
                n_trees       = 100
                learning_rate = 0.2 
                max_depth     = 2
                min_size      = 50

                bit = BoostedInformationTree(
                        training_features = training_features,
                        training_weights      = training_weights, 
                        training_diff_weights = training_diff_weights, 
                        learning_rate = learning_rate, 
                        n_trees = n_trees,
                        max_depth=max_depth,
                        min_size=min_size,
                        split_method='vectorized_split_and_weight_sums',
                        weights_update_method='vectorized')

                bit.boost()

                # unvectorized predictions
                predicted_scores_20 = [bit.predict(features, max_n_tree=20, vectorized=False) for features in training_features]
                
                time1 = time.time()
                predicted_scores_100 = [bit.predict(features,  max_n_tree=100, vectorized=False) for features in training_features]
                time2 = time.time()
                print "prediction time unvec: %.2f seconds" % (time2 - time1)

                training_FI_20 = np.dot(training_diff_weights, predicted_scores_20)
                print "training FI_20: %f" % training_FI_20
                self.assertAlmostEqual(278353.986701, training_FI_20)

                training_FI_100 = np.dot(training_diff_weights, predicted_scores_100)
                print "training FI_100: %f" % training_FI_100
                self.assertAlmostEqual(284127.8915964808, training_FI_100)

                # now the same predictions vectorized
                predicted_scores_20 = bit.predict(training_features, max_n_tree=20, vectorized=True)
                
                time1 = time.time()
                predicted_scores_100 = bit.predict(training_features,  max_n_tree=100, vectorized=True)
                time2 = time.time()
                print "prediction time vec: %.2f seconds" % (time2 - time1)

                training_FI_20 = np.dot(training_diff_weights, predicted_scores_20)
                print "training FI_20: %f" % training_FI_20
                #self.assertAlmostEqual(278353.986701, training_FI_20)

                training_FI_100 = np.dot(training_diff_weights, predicted_scores_100)
                print "training FI_100: %f" % training_FI_100
                #self.assertAlmostEqual(284127.8915964808, training_FI_100)

if __name__ == '__main__':
    unittest.main()