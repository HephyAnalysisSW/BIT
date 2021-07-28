#!/usr/bin/env python

import unittest
import numpy as np
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
                print(weights.shape)

                min_size = 10000

                assert len(features)==len(weights), "Need equal length for weights and features."

                #FI_func = lambda coeffs: w.get_fisherInformation_matrix( coeffs, variables = ['cWWW'], cWWW=1)[1][0][0]
                weight_mask = w.get_weight_mask( cWWW=1 )
                diff_weight_mask = w.get_diff_mask( 'cWWW', cWWW=1 )

                #TODO:
                training_weights         = np.dot(weights, w.get_weight_mask(cWWW=1))
                training_diff_weights    = np.dot(weights, w.get_diff_mask('cWWW', cWWW=1))

if __name__ == '__main__':
    unittest.main()
