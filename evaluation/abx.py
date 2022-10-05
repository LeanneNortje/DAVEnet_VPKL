#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import ast
import logging
import numexpr
import numpy as np
import os
import pandas
import shutil
import sys
import tempfile
import warnings

import ABXpy
from ABXpy.distance import default_distance, dtw_kl_distance, edit_distance
from ABXpy.misc.any2h5features import convert
from ABXpy.score import score
from ABXpy.analyze import analyze

def abx(features_path, temp_dir, task, task_type, load_fun,
         distance, normalized, njobs, log):
    """Runs the ABX pipeline"""
    dist2fun = {
        'cosine': default_distance,
        'KL': dtw_kl_distance,
        'levenshtein': edit_distance}

    # convert
    log.debug('loading features ...')
    features = os.path.join(temp_dir, 'features.h5')
    if not os.path.isfile(features):
        convert(features_path, h5_filename=features, load=load_fun)

    # avoid annoying log message
    numexpr.set_num_threads(njobs)

    log.debug('computing %s distances ...', distance)
    # ABX Distances prints some messages we do not want to display
    sys.stdout = open(os.devnull, 'w')
    distance_file = os.path.join(temp_dir, 'distance_{}.h5'.format(task_type))
    with warnings.catch_warnings():
        # inhibit some useless warnings about complex to float conversion
        warnings.filterwarnings("ignore", category=np.ComplexWarning)

        # compute the distances
        ABXpy.distances.distances.compute_distances(
            features,
            'features',
            task,
            distance_file,
            dist2fun[distance],
            normalized,
            n_cpu=njobs)
    sys.stdout = sys.__stdout__

    log.debug('computing abx score ...')
    # score
    score_file = os.path.join(temp_dir, 'score_{}.h5'.format(task_type))
    score(task, distance_file, score_file)

    # analyze
    analyze_file = os.path.join(temp_dir, 'analyze_{}.csv'.format(task_type))
    analyze(task, score_file, analyze_file)

    # average
    abx_score = _average(analyze_file, task_type)
    return abx_score