"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a Support Vector Machine (SVM) classifier.
"""
import argparse
import datetime
import os
import sys
import timeit

import SimpleITK as sitk
import numpy as np
from tensorflow.python.platform import app

sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing main.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

import mialab.data.conversion as conversion
import mialab.data.structure as structure
import mialab.data.loading as load
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil
import mialab.utilities.statistic_utilities as statistics

from sklearn.svm import SVC
from scipy import stats as scipy_stats
import sklearn.preprocessing as sk_preprocessing
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import scipy.io


FLAGS = None  # the program flags
IMAGE_KEYS = [structure.BrainImageTypes.T1, structure.BrainImageTypes.T2, structure.BrainImageTypes.GroundTruth]  # the list of images we will load
TEST_BATCH_SIZE = 2  # 1..30, the higher the faster but more memory usage
NORMALIZE_FEATURES = False # Normalize feature vectors to mean 0 and std 1

# Utility function to move the midpoint of a colormap to be around
# the values of interest.
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def main(_):
    """Brain tissue segmentation using SVM.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - SVM model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # SVM cannot deal with default mark (too much data). Reduce by factor 10
    putil.FeatureExtractor.VOXEL_MASK_FLT = [0.00003, 0.0004, 0.0003, 0.0004]

    # load atlas images
    putil.load_atlas_images(FLAGS.data_atlas_dir)

    print('-' * 5, 'Training...')

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
    train_data_size = len(data_items)

    pre_process_params = {'zscore_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    start_time_total_train = timeit.default_timer()

    batch_data = dict(data_items)
    # load images for training and pre-process
    images = putil.pre_process_batch(batch_data, pre_process_params, multi_process=True)
    print('pre-processing done')

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images])

    if NORMALIZE_FEATURES:
        # normalize data (mean 0, std 1)
        data_train = scipy_stats.zscore(data_train)

    print('Start training SVM')

    # Training
    # SVM does not support online/incremental training. Need to fit all in one go!
    # Note: Very slow with large training set!
    start_time = timeit.default_timer()
    # to limite: max_iter=1000000000

    # Enable for grid search of best hyperparameters
    if False:
        C_range = [300, 350, 400, 450, 500, 550, 600, 800, 1000, 1200, 1500]
        gamma_range = [0.00001, 0.00003, 0.00004, 0.00005, 0.00006, 0.00008, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.2]

        # 1
        C_range = [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 10, 20, 50, 100, 200, 250, 300, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 120000, 150000]
        gamma_range = [0.0000001, 0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.2, 0.5, 1, 5, 10]

        #C_range = [1, 10, 100, 500, 1000, 5000, 10000, 15000, 20000, 22000, 25000, 30000, 35000]
        #gamma_range = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.2, 0.5]

        params = [{
                'kernel': ['rbf'],
                'C': C_range,
                'gamma': gamma_range,
        }]
        #'C': [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 10, 20, 50, 100, 200, 250, 300, 1000],
        #'gamma': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.2, 0.5, 1, 5, 10, 20, 100, 10

        clf = GridSearchCV(SVC(probability=True, cache_size=2000), params, cv=2, n_jobs=8, verbose=3)
        clf.fit(data_train, labels_train[:, 0])
        print('best param: ' + str(clf.best_params_))
        scores = clf.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.savefig('svm_params.png')
        #plt.show()

        scipy.io.savemat('svm_params.mat', mdict={'C': C_range, 'gamma': gamma_range, 'score': scores})

    #svm = SVC(probability=True, kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], cache_size=2000, verbose=False)

    svm = SVC(probability=True, kernel='rbf', C=500, gamma=0.00005, cache_size=2000, verbose=False)

    svm.fit(data_train, labels_train[:, 0])
    print('\n Time elapsed:', timeit.default_timer() - start_time, 's')
    time_total_train = timeit.default_timer() - start_time_total_train

    start_time_total_test = timeit.default_timer()
    print('-' * 5, 'Testing...')
    result_dir = os.path.join(FLAGS.result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    # initialize evaluator
    evaluator = putil.init_evaluator(result_dir)

    # crawl the training image directories
    crawler = load.FileSystemDataCrawler(FLAGS.data_test_dir,
                                         IMAGE_KEYS,
                                         futil.BrainImageFilePathGenerator(),
                                         futil.DataDirectoryFilter())
    data_items = list(crawler.data.items())

    all_probabilities = None

    for batch_index in range(0, len(data_items), TEST_BATCH_SIZE):
        # slicing manages out of range; no need to worry
        batch_data = dict(data_items[batch_index: batch_index + TEST_BATCH_SIZE])

        # load images for testing and pre-process
        pre_process_params['training'] = False
        images_test = putil.pre_process_batch(batch_data, pre_process_params, multi_process=True)

        images_prediction = []
        images_probabilities = []

        for img in images_test:
            print('-' * 10, 'Testing', img.id_)

            start_time = timeit.default_timer()
            #probabilities, predictions = forest.predict(img.feature_matrix[0])
            features = img.feature_matrix[0]
            if NORMALIZE_FEATURES:
                features = scipy_stats.zscore(features)
            probabilities = np.array(svm.predict_proba(features))
            print('probabilities: ' + str(probabilities.shape))
            predictions = svm.classes_[probabilities.argmax(axis=1)]

            if all_probabilities is None:
                all_probabilities = np.array([probabilities])
            else:
                all_probabilities = np.concatenate((all_probabilities, [probabilities]), axis=0)

            print(' Time elapsed:', timeit.default_timer() - start_time, 's')

            # convert prediction and probabilities back to SimpleITK images
            image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                            img.image_properties)
            image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

            # evaluate segmentation without post-processing
            evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

            images_prediction.append(image_prediction)
            images_probabilities.append(image_probabilities)

        # post-process segmentation and evaluate with post-processing
        post_process_params = {'crf_post': True}
        images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                         post_process_params, multi_process=True)

        for i, img in enumerate(images_test):
            evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                               img.id_ + '-PP')

            # save results
            sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
            sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)

    time_total_test = timeit.default_timer() - start_time_total_test

    # write summary of parameters to results dir
    with open(os.path.join(result_dir, 'summary.txt'), 'w') as summary_file:
        print('Result dir: {}'.format(result_dir))
        print('Result dir: {}'.format(result_dir), file=summary_file)
        print('SVM', file=summary_file)
        print('SVM params: {}'.format(svm.get_params()), file=summary_file)
        print('pre-process-params: {}'.format(pre_process_params), file=summary_file)
        print('Training data size: {}'.format(train_data_size), file=summary_file)
        print('Total training time: {:.1f}s'.format(time_total_train), file=summary_file)
        print('Total testing time: {:.1f}s'.format(time_total_test), file=summary_file)
        print('Voxel Filter Mask: {}'.format(putil.FeatureExtractor.VOXEL_MASK_FLT), file=summary_file)
        print('Normalize Features: {}'.format(NORMALIZE_FEATURES), file=summary_file)
        #print('SVM best parameters', file=summary_file)
        #print(clf.best_params_, file=summary_file)
        stats = statistics.gather_statistics(os.path.join(result_dir, 'results.csv'))
        print('Result statistics:', file=summary_file)
        print(stats, file=summary_file)

    # all_probabilities.astype(np.float16).dump(os.path.join(result_dir, 'all_probabilities.npy'))


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
