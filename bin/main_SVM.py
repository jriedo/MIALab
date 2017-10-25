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


FLAGS = None  # the program flags
IMAGE_KEYS = [structure.BrainImageTypes.T1, structure.BrainImageTypes.T2, structure.BrainImageTypes.GroundTruth]  # the list of images we will load
TEST_BATCH_SIZE = 2  # 1..30, the higher the faster but more memory usage
NORMALIZE_FEATURES = False # Normalize feature vectors to mean 0 and std 1

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
    svm = SVC(probability=True, kernel='linear', C=1, cache_size=2000, verbose=False)

    svm.fit(data_train, labels_train[:, 0])
    print('\n Time elapsed:', timeit.default_timer() - start_time, 's')

    time_total_train = timeit.default_timer() - start_time_total_train
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


    # write summary of parameters to results dir
    with open(os.path.join(result_dir, 'summary.txt'), 'w') as summary_file:
        print('Training data size: {}'.format(train_data_size), file=summary_file)
        print('Total training time: {:.1f}s'.format(time_total_train), file=summary_file)
        print('Voxel Filter Mask: {}'.format(putil.FeatureExtractor.VOXEL_MASK_FLT), file=summary_file)
        print('Normalize Features: {}'.format(NORMALIZE_FEATURES), file=summary_file)
        #print('SVM best parameters', file=summary_file)
        #print(clf.best_params_, file=summary_file)
        stats = statistics.gather_statistics(os.path.join(result_dir, 'results.csv'))
        print('Result statistics:', file=summary_file)
        print(stats, file=summary_file)


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
