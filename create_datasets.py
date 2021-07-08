#!/usr/bin/env python
"""Create test and training datasets from different toy models."""

import numpy as np
from toy_models import exponential, power_law, piece_wise

n_events = 100000
data_dir = 'data'

def create_and_save_datasets(name, model, n_events):
    for dataset_type in ['training', 'test']:
        features, weights, diff_weights = model.get_dataset(n_events)
        np.savetxt('%s/%s_features_%s.txt.gz' % (data_dir, dataset_type, name), features)
        np.savetxt('%s/%s_weights_%s.txt.gz' % (data_dir, dataset_type, name), weights)
        np.savetxt('%s/%s_diff_weights_%s.txt.gz' % (data_dir, dataset_type, name), diff_weights)

create_and_save_datasets('exponential_model', exponential, n_events)
create_and_save_datasets('power_law_model', power_law, n_events)
create_and_save_datasets('piece_wise_model', piece_wise, n_events)
