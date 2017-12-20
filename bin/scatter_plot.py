"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys

import numpy as np
from tensorflow.python.platform import app

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing main.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

import mialab.data.structure as structure
import mialab.data.loading as load
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil

FLAGS = None  # the program flags
IMAGE_KEYS = [structure.BrainImageTypes.T1, structure.BrainImageTypes.T2, structure.BrainImageTypes.GroundTruth]  # the list of images we will load
TRAIN_BATCH_SIZE = 70  # 1..70, the higher the faster but more memory usage
TEST_BATCH_SIZE = 30 # 1..30, the higher the faster but more memory usage
USE_PREPROCESS_CACHE = False    # cache pre-processed images


def main(_):
    # generate a model directory (use datetime to ensure that the directory is empty)
    # we need an empty directory because TensorFlow will continue training an existing model if it is not empty
    t = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
    model_dir = os.path.join(FLAGS.model_dir, t)
    os.makedirs(model_dir, exist_ok=True)

    # crawl the training image directories
    crawler = load.FileSystemDataCrawler(FLAGS.data_train_dir,
                                         IMAGE_KEYS,
                                         futil.BrainImageFilePathGenerator(),
                                         futil.DataDirectoryFilter())
    data_items = list(crawler.data.items())

    pre_process_params = {'zscore_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    for batch_index in range(0, len(data_items), TRAIN_BATCH_SIZE):
        cache_file_prefix = os.path.normpath(os.path.join(script_dir, './mia-cache/batch-' + str(batch_index) + '-' + str(TRAIN_BATCH_SIZE)))
        cache_file_train = cache_file_prefix + '-data_train.npy'
        cache_file_labels = cache_file_prefix + '-data_labels.npy'
        if(USE_PREPROCESS_CACHE & os.path.exists(cache_file_train)):
            print('Using cache from ', cache_file_train)
            data_train = np.load(cache_file_train)
            labels_train = np.load(cache_file_labels)
        else:
            # slicing manages out of range; no need to worry
            batch_data = dict(data_items[batch_index: batch_index+TRAIN_BATCH_SIZE])
            # load images for training and pre-process
            images = putil.pre_process_batch(batch_data, pre_process_params, multi_process=True)

            # generate feature matrix and label vector
            data_train = np.concatenate([img.feature_matrix[0] for img in images])
            labels_train = np.concatenate([img.feature_matrix[1] for img in images])


    # Scatter matrix plot of the train data

    data = pd.DataFrame(data_train, columns=['Feat. 1', 'Feat. 2', 'Feat. 3', 'Feat. 4', 'Feat. 5',
                                                                                         'Feat. 6', 'Feat. 7'])
    axes = pd.scatter_matrix(data, alpha=0.2, diagonal='hist')
    corr = data.corr().as_matrix()
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        axes[i, j].annotate("%.2f" % corr[i, j], (0.99, 0.98), size=23, xycoords='axes fraction', ha='right', va='top')

    n = len(data.columns)
    for x in range(n):
        for y in range(n):
            # to get the axis of subplots
            ax = axes[x, y]
            # to make x axis name vertical
            ax.xaxis.label.set_rotation(0)
            ax.xaxis.label.set_size(17)
            ax.xaxis.set_label_coords(0.5, -0.3)
            # to make y axis name horizontal
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_size(17)
            ax.yaxis.set_label_coords(-0.3, 0.5)
            # to make sure y axis names are outside the plot area
            ax.yaxis.labelpad = 50

    # plt.title('Scatter Plot Matrix', fontsize=17, y=7.1, x=-2.5)
    plt.show()

if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-model')),
        help='Base directory for output models.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
