import   numpy as np
import   ROOT
import   array


def clip_quantile( features, quantile, weights = None ):

    selected  = np.array(list(range(len(features)))).reshape(-1)
    selection = np.ones_like( selected ).astype('bool')

    for i_feature in range(len(features[0])):
        selection&= 1==np.digitize( features[:, i_feature], np.quantile( features[:, i_feature], ( quantile, 1.-quantile )) )

    #len_before = len(selected)
    selected = selected[selection]
    #print( "Autoclean efficiency of %3.2f: %3.2f"%(args.auto_clean, np.count_nonzero( selection )/len_before) )
    return_features = features[selected]
    if weights is not None:
        return_weights = {k:weights[k][selected] for k in weights.keys()}
        return return_features, return_weights
    else:
        return return_features

def make_TH1F( h, ignore_binning = False):
    # remove infs from thresholds
    vals, thrs = h
    if ignore_binning:
        histo = ROOT.TH1F("h","h",len(vals),0,len(vals))
    else:
        histo = ROOT.TH1F("h","h",len(thrs)-1,array.array('d', thrs))
    for i_v, v in enumerate(vals):
        if v<float('inf'): # NAN protection
            histo.SetBinContent(i_v+1, v)
    return histo

# https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

import os, shutil
def copyIndexPHP( directory ):
    ''' Copy index.php to directory
    '''
    index_php = os.path.join( directory, 'index.php' )
    if not os.path.exists( directory ): os.makedirs( directory )
    shutil.copyfile( os.path.join(os.path.dirname(__file__), 'scripts/index.php'), index_php )
