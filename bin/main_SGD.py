"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
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


# pip install sklearn
# pip install scipy

import mialab.classifier.decision_forest as df
import mialab.data.conversion as conversion
import mialab.data.structure as structure
import mialab.data.loading as load
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil
import mialab.utilities.statistic_utilities as statistics

#import sklearn
from sklearn import cross_validation, grid_search
#from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from scipy import stats as scipy_stats

FLAGS = None  # the program flags
IMAGE_KEYS = [structure.BrainImageTypes.T1, structure.BrainImageTypes.T2, structure.BrainImageTypes.GroundTruth]  # the list of images we will load
TRAIN_BATCH_SIZE = 70  # 1..70, the higher the faster but more memory usage
TEST_BATCH_SIZE = 2  # 1..30, the higher the faster but more memory usage
USE_PREPROCESS_CACHE = False    # cache pre-processed images
NORMALIZE_FEATURES = False # Normalize feature vectors to mean 0 and std 1

def main(_):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # SGD need "original" value of 0.04 for ventricles
    putil.FeatureExtractor.VOXEL_MASK_FLT = [0.0003, 0.004, 0.003, 0.04]

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

    # initialize decision forest parameters
    df_params = df.DecisionForestParameters()
    df_params.num_classes = 4
    df_params.num_trees = 20
    df_params.max_nodes = 1000
    df_params.model_dir = model_dir
    forest = None
    clf = None
    start_time_total_train = timeit.default_timer()

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
            print('pre-processing done')

            # generate feature matrix and label vector
            data_train = np.concatenate([img.feature_matrix[0] for img in images])
            labels_train = np.concatenate([img.feature_matrix[1] for img in images])

            if NORMALIZE_FEATURES:
                # normalize data (mean 0, std 1)
                data_train = scipy_stats.zscore(data_train)

            if(USE_PREPROCESS_CACHE):
                print('Writing cache')
                if(not os.path.exists(os.path.dirname(cache_file_prefix))):
                    os.mkdir(os.path.dirname(cache_file_prefix))
                data_train.dump(cache_file_train)
                labels_train.dump(cache_file_labels)


        if clf is None:
            # cross validation to find best parameter
            param = [
                {
                    "eta0": [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                    "alpha": [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                    "learning_rate": ['optimal', 'constant'],
                    "loss": ['log', 'modified_huber']
                    #"max_iter": [10000, 100000]
                },
            ]
            # Best params:
            #{'alpha': 0.01, 'eta0': 0.5, 'learning_rate': 'optimal', 'loss': 'modified_huber'}

            sgd = SGDClassifier(learning_rate = 'optimal', eta0 =0.5, alpha=0.01,
                               loss='modified_huber', penalty="l2", max_iter=100000, n_jobs=8, shuffle=False)
            clf = sgd
            # Note: shuffle=True gives '"RuntimeWarning: overflow encountered in expnp.exp(prob, prob)"'

            # to try several parameters with grid search
            #clf = grid_search.GridSearchCV(sgd, param, cv=2, n_jobs=4, verbose=3)

        start_time = timeit.default_timer()

        clf.fit(data_train, labels_train[:, 0])
        #print('Best params: ')
        #print(clf.best_params_)
        print('\n training, Time elapsed:', timeit.default_timer() - start_time, 's')

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
            probabilities = np.array(clf.predict_proba(features))
            print('probabilities: ' + str(probabilities.shape))
            predictions = clf.classes_[probabilities.argmax(axis=1)]

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
        print('Training data size: {}'.format(train_data_size), file=summary_file)
        print('Total training time: {:.1f}s'.format(time_total_train), file=summary_file)
        print('Total testing time: {:.1f}s'.format(time_total_test), file=summary_file)
        print('Voxel Filter Mask: {}'.format(putil.FeatureExtractor.VOXEL_MASK_FLT), file=summary_file)
        print('Normalize Features: {}'.format(NORMALIZE_FEATURES), file=summary_file)
        print('SGD', file=summary_file)
        #print(clf.best_params_, file=summary_file)
        stats = statistics.gather_statistics(os.path.join(result_dir, 'results.csv'))
        print('Result statistics:', file=summary_file)
        print(stats, file=summary_file)

    # all_probabilities.astype(np.float32).dump(os.path.join(result_dir, 'all_probabilities.npy'))



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
