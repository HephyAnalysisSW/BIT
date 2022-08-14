#!/usr/bin/env python

import unittest
import numpy as np
import pandas as pd
import time
import uproot

class TestWeightInfo(unittest.TestCase):
        # test weight diffs unvectorized vs vectorized
        def test_first_and_second_derivatives(self):
                max_events  = 10000
                input_file  = "/eos/vbc/user/robert.schoefbeck/TMB/bit/MVA-training/ttG_WG_small/WGToLNu_fast/WGToLNu_fast.root"
                #input_file  = "/scratch-cbe/users/nikolaus.frohner/TMB/bit/MVA-training/ttG_WG/WGToLNu_fast/WGToLNu_fast.root"
                upfile      = uproot.open( input_file )
                tree        = upfile["Events"]
                n_events    = len( upfile["Events"] )
                n_events    = min(max_events, n_events)
                entrystart, entrystop = 0, n_events 

                # Load features
                #branches    = [ "mva_photon_pt", ]#"mva_photon_eta", "mva_photonJetdR", "mva_photonLepdR", "mva_mT" ]
                branches    = [ "mva_photon_pt" , "mva_photon_eta", "mva_photonJetdR", "mva_photonLepdR", "mva_mT" ]
                df          = tree.pandas.df(branches = branches, entrystart=entrystart, entrystop=entrystop)
                features    = df.values

                print(features.shape)
                from WeightInfo import WeightInfo
                w = WeightInfo("/eos/vbc/user/robert.schoefbeck/gridpacks/v6/WGToLNu_reweight_card.pkl")
                w.set_order(2)

                # Load all weights and reshape the array according to ndof from weightInfo
                weights     = tree.pandas.df(branches = ["p_C"], entrystart=entrystart, entrystop=entrystop).values.reshape((-1,w.nid))
                #df_weights = tree.pandas.df(branches = ["p_C"], entrystart=entrystart, entrystop=entrystop)
                print(weights.shape)

                min_size = 10000

                assert len(features)==len(weights), "Need equal length for weights and features."

                time_weights_1 = time.time()
                weight_mask = w.get_weight_mask( cWWW=1 )
                training_weights         = np.dot(weights, w.get_weight_mask(cWWW=1))
                time_weights_2 = time.time()
                print "weights calc time vec: %.4f seconds" % (time_weights_2 - time_weights_1)

                # unvectorized diff weights
                time_unvec_diff_weights_1 = time.time()
                diff_weight_calculator = w.get_diff_weight_func('cWWW', cWWW=1)
                unvec_training_diff_weights = np.apply_along_axis(diff_weight_calculator, axis=1, arr=weights)
                time_unvec_diff_weights_2 = time.time()
                print(unvec_training_diff_weights.shape)
                print "diff weights calc time uvec: %.4f seconds" % (time_unvec_diff_weights_2 - time_unvec_diff_weights_1)

                # vectorized diff weights
                time_vec_diff_weights_1 = time.time()
                diff_weight_mask = w.get_diff_mask( 'cWWW', cWWW=1 )
                training_diff_weights    = np.dot(weights, diff_weight_mask)
                time_vec_diff_weights_2 = time.time()
                print(training_diff_weights.shape)
                print "diff weights calc time vec: %.4f seconds" % (time_vec_diff_weights_2 - time_vec_diff_weights_1)

                np.testing.assert_almost_equal(training_diff_weights, unvec_training_diff_weights)

                # unvectorized double diff weights
                time_unvec_double_diff_weights_1 = time.time()
                double_diff_weight_calculator = w.get_double_diff_weight_func('cWWW', cWWW=1)
                unvec_training_double_diff_weights = np.apply_along_axis(double_diff_weight_calculator, axis=1, arr=weights)
                time_unvec_double_diff_weights_2 = time.time()
                print(unvec_training_double_diff_weights.shape)
                print "double diff weights calc time unvec: %.4f seconds" % (time_unvec_double_diff_weights_2 - time_unvec_double_diff_weights_1)

                # vectorized double diff weights
                time_vec_diff_weights_1 = time.time()
                double_diff_weight_mask = w.get_double_diff_mask( 'cWWW', cWWW=1 )
                training_double_diff_weights    = np.dot(weights, double_diff_weight_mask)
                time_vec_diff_weights_2 = time.time()
                print(training_double_diff_weights.shape)
                print "double diff weights calc time vec: %.4f seconds" % (time_vec_diff_weights_2 - time_vec_diff_weights_1)

                np.testing.assert_almost_equal(training_double_diff_weights, unvec_training_double_diff_weights)


if __name__ == '__main__':
    unittest.main()
