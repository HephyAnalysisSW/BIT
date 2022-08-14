#!/usr/bin/env python

import unittest
import numpy as np
import time
import sys
sys.path.insert(0,'..')
from BoostedInformationTreeP3 import BoostedInformationTree
from NodeP3 import Node
class TestBIT(unittest.TestCase):
        def test_power_law_model(self):
                data_dir = 'data'

                training_features = np.loadtxt('../%s/training_features_power_law_model.txt.gz' % data_dir)
                training_features = training_features.reshape(training_features.shape[0], -1)
                print(training_features.shape)
                training_weights = np.loadtxt('../%s/training_weights_power_law_model.txt.gz' % data_dir)
                training_diff_weights = np.loadtxt('../%s/training_diff_weights_power_law_model.txt.gz' % data_dir)

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
                predicted_scores_20 = [bit.predict(features, max_n_tree=20) for features in training_features]
                
                time1 = time.time()
                predicted_scores_100 = [bit.predict(features,  max_n_tree=100) for features in training_features]
                time2 = time.time()
                print("prediction time unvec: %.2f seconds" % (time2 - time1))

                training_FI_20 = np.dot(training_diff_weights, predicted_scores_20)
                print("training FI_20: %f" % training_FI_20)
                self.assertAlmostEqual(278353.1436750, training_FI_20)

                training_FI_100 = np.dot(training_diff_weights, predicted_scores_100)
                print("training FI_100: %f" % training_FI_100)
                self.assertAlmostEqual(284138.66016375029, training_FI_100)

                # now the same predictions vectorized
                predicted_scores_20 = bit.vectorized_predict(training_features, max_n_tree=20)
                
                time1 = time.time()
                predicted_scores_100 = bit.vectorized_predict(training_features,  max_n_tree=100)
                time2 = time.time()
                print("prediction time vec: %.2f seconds" % (time2 - time1))

                training_FI_20 = np.dot(training_diff_weights, predicted_scores_20)
                print("training FI_20: %f" % training_FI_20)
                self.assertAlmostEqual(278353.1436750, training_FI_20)

                training_FI_100 = np.dot(training_diff_weights, predicted_scores_100)
                print("training FI_100: %f" % training_FI_100)
                self.assertAlmostEqual(284138.66016375029, training_FI_100)

        def test_node_splitting(self):
                features = np.arange(10).reshape(-1, 1)
                weights = np.ones(10).reshape(-1, 1)
                weight_diffs = np.array([1,1,5,1,1,1,1,10,1,1]).reshape(-1, 1)
                max_depth = 2
                min_size = 3

                weak_learner = Node( features, max_depth=max_depth, min_size=min_size, training_weights=weights, training_diff_weights=weight_diffs, split_method='vectorized_split_and_weight_sums' )
                self.assertEqual(weak_learner.split_value, 6)
        
        def test_power_law_model_with_calibration(self):
                data_dir = 'data'

                training_features = np.loadtxt('../%s/training_features_power_law_model.txt.gz' % data_dir)
                training_features = training_features.reshape(training_features.shape[0], -1)
                print(training_features.shape)
                training_weights = np.loadtxt('../%s/training_weights_power_law_model.txt.gz' % data_dir)
                training_diff_weights = np.loadtxt('../%s/training_diff_weights_power_law_model.txt.gz' % data_dir)

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
                        calibrated=True,
                        split_method='vectorized_split_and_weight_sums',
                        weights_update_method='vectorized')

                bit.boost()
                # unvectorized predictions
                predicted_scores_20 = [bit.predict(features, max_n_tree=20) for features in training_features]
                self.assertAlmostEqual(82886.331011181421, np.sum(predicted_scores_20))
                
                predicted_scores_20 = bit.vectorized_predict(training_features, max_n_tree=20)
                self.assertAlmostEqual(82886.331011181421, np.sum(predicted_scores_20))

if __name__ == '__main__':
    unittest.main()
