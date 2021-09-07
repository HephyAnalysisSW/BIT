import ROOT
from RootTools.core.standard import *
import os

def make_debug_plots( bit, test_features, test_weights, test_diff_weights, plot_directory):

    for n_tree, tree in enumerate( bit.trees ):

        # debug data for tree n_tree
        features        = bit.debug_data[n_tree]["features"]
        weights         = bit.debug_data[n_tree]["weights"]
        diff_weights    = bit.debug_data[n_tree]["diff_weights"]
        features        = bit.debug_data[n_tree]["features"]

        split_i_feature = bit.debug_data[n_tree]["split_i_feature"]
        split_value     = bit.debug_data[n_tree]["split_value"]

        h_der = ROOT.TH1D("h_derivative_%i"%n_tree, "feature %i"%split_i_feature, 50, min(features), max(features))
        h_wgt = ROOT.TH1D("h_weight_%i"%n_tree, "feature%i"%split_i_feature, 50, min(features), max(features))

        h_bit = ROOT.TH1D("h_predicted_%i"%n_tree, "feature %i"%split_i_feature, 50, min(features), max(features))

        for feature, weight, derivative in zip(features, weights, diff_weights):
            h_der.Fill( feature, derivative )
            h_wgt.Fill( feature, weight )

            h_bit.Fill( feature, weight*bit.predict([feature], max_n_tree=n_tree) )

        h_der_test = ROOT.TH1D("h_derivative_test_%i"%n_tree, "feature %i"%split_i_feature, 50, min(features), max(features))
        h_der_test.style = styles.lineStyle(ROOT.kBlue)
        for feature, weight, derivative in zip(test_features[:,split_i_feature], test_weights, test_diff_weights):
            h_der_test.Fill( feature, derivative )

        # recall the starting point:
        if n_tree == 0:
            h_der_start = h_der.Clone("h_der_start")
            h_der_start.style = styles.lineStyle(ROOT.kRed, dashed=True)

        h_scr = h_der.Clone("h_score_%i"%n_tree)
        h_scr.Divide(h_wgt)

        h_der.style = styles.lineStyle(ROOT.kRed, dashed=False)
        h_bit.style = styles.lineStyle(ROOT.kGreen+1, width=2)

        h_der_start.legendText = "#sum w' initial"
        h_der_test .legendText = "#sum w' initial (test)" 
        h_der      .legendText = "#sum w' step %i"%n_tree
        h_bit      .legendText = "#sum( w#times t(x)) step %i"%n_tree

        plot = Plot.fromHisto("der_%03i"%n_tree, [[h_der_start], [h_der_test], [h_der], [h_bit]], texX = "feature %i"%split_i_feature, texY = "#sum(w')")

        cut_line = ROOT.TLine( tree.split_value, h_der_start.GetMinimum(), split_value, h_der_start.GetMaximum() )
        cut_line.SetLineColor(ROOT.kRed)
        cut_line.SetLineWidth(2)

        plotting.draw( plot, plot_directory = os.path.join( plot_directory, "lin" ), drawObjects = [cut_line], copyIndexPHP=True, logY = False)
        #plot.name = "der"
        #plotting.draw( plot, plot_directory = os.path.join( plot_directory, "lin" ), drawObjects = [cut_line], copyIndexPHP=True, logY = False, extensions = ["gif+"])


