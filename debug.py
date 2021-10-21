import ROOT
from RootTools.core.standard import *
import os

def make_debug_plots( bit, training_features, training_weights, training_diff_weights, test_features, test_weights, test_diff_weights, plot_directory, mva_variables = None):

    # where we start the training
    h_der_start = {}
    for i_feature in range(len(training_features[0])):
        min_, max_ = min(training_features[:,i_feature]), max(training_features[:,i_feature])
        h_der_start[i_feature] = ROOT.TH1D("h_derivative_start", "feature %i"%i_feature, 50, min_, max_)
        h_der_start[i_feature].style = styles.lineStyle(ROOT.kRed, dashed=True)
        h_der_start[i_feature].legendText = "#sum w' initial"

    for n_tree, tree in enumerate( bit.trees ):

        #'features':      np.copy(self.training_features[bagging_mask][:, root.split_i_feature]),
        #'weights' :      np.copy(self.training_weights[bagging_mask]),
        #'diff_weights' : np.copy(self.training_diff_weights[bagging_mask]),

        # debug data for tree n_tree
        split_i_feature = bit.debug_data[n_tree]["split_i_feature"]
        split_value     = bit.debug_data[n_tree]["split_value"]
        features        = training_features[bit.debug_data[n_tree]["mask"]]
        feature_values  = features[:, split_i_feature]
        #weights         = training_weights[bit.debug_data[n_tree]["mask"]]
        #diff_weights    = training_diff_weights[bit.debug_data[n_tree]["mask"]]
        weights         =  bit.debug_data[n_tree]["weights"] 
        diff_weights    =  bit.debug_data[n_tree]["diff_weights"] 


        min_, max_ = min(feature_values), max(feature_values)
        h_der = ROOT.TH1D("h_derivative_%i"%n_tree, "feature %i"%split_i_feature, 50, min_, max_)
        h_wgt = ROOT.TH1D("h_weight_%i"%n_tree,     "feature %i"%split_i_feature, 50, min_, max_)

        h_bit = ROOT.TH1D("h_predicted_%i"%n_tree,  "feature %i"%split_i_feature, 50, min_, max_)

        for feature, weight, derivative in zip(features, weights, diff_weights):
            h_der.Fill( feature[split_i_feature], derivative )
            h_wgt.Fill( feature[split_i_feature], weight )
            #print feature, weight, derivative
            h_bit.Fill( feature[split_i_feature], weight*bit.predict(feature, max_n_tree=n_tree,add_global_score=False) )
            # recall the starting point:
            if n_tree == 0:
                for i_feature, v_feature in enumerate(feature):
                    h_der_start[i_feature].Fill( v_feature, derivative )

        h_der_test = ROOT.TH1D("h_derivative_test_%i"%n_tree, "feature %i"%split_i_feature, 50, min_, max_)
        h_der_test.style = styles.lineStyle(ROOT.kBlue)

        global_score = bit.global_score if hasattr( bit, "global_score") else 0
        for feature, weight, derivative in zip(test_features[:,split_i_feature], test_weights, test_diff_weights):
            h_der_test.Fill( feature, derivative - weight*global_score )

        h_scr = h_der.Clone("h_score_%i"%n_tree)
        h_scr.Divide(h_wgt)

        h_der.style = styles.lineStyle(ROOT.kRed, dashed=False)
        h_bit.style = styles.lineStyle(ROOT.kGreen+1, width=2)

        h_der_test .legendText = "#sum w' initial (test)" 
        h_der      .legendText = "#sum w' step %i"%n_tree
        h_bit      .legendText = "#sum( w#times t(x)) step %i"%n_tree

        plot = Plot.fromHisto("der_%03i"%n_tree, [[h_der_start[split_i_feature]], [h_der_test], [h_der], [h_bit]], 
            texX = "feature %i"%split_i_feature if mva_variables is None else mva_variables[split_i_feature][0], 
            texY = "#sum(w')")

        cut_line = ROOT.TLine( tree.split_value, h_der_start[split_i_feature].GetMinimum(), split_value, h_der_start[split_i_feature].GetMaximum() )

        cut_line.SetLineColor(ROOT.kRed)
        cut_line.SetLineWidth(2)

        plotting.draw( plot, plot_directory = os.path.join( plot_directory, "lin" ), drawObjects = [cut_line], copyIndexPHP=True, logY = False)
