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

import mialab.data.conversion as conversion
import mialab.data.structure as structure
import mialab.data.loading as load
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil
import mialab.utilities.statistic_utilities as statistics

FLAGS = None  # the program flags
IMAGE_KEYS = [structure.BrainImageTypes.T1, structure.BrainImageTypes.T2, structure.BrainImageTypes.GroundTruth]  # the list of images we will load
TEST_BATCH_SIZE = 2  # 1..30, the higher the faster but more memory usage
LABEL_CLASSES = np.array([0, 1, 2, 3])
ENSEMBLE_MAX = False     # True: use max probability. False: Average probabilities
RESULTS = ['bin/mia-result/2017-10-28080136',
           'bin/mia-result/2017-10-28080131',
           'bin/mia-result/2017-10-28080133',
           'bin/mia-result/2017-10-28080134']

def main(_):
    """Ensemble using results from various algorithms
    """

    # load results from various previous runs
    all_probabilities = None
    for r in RESULTS:
        p = np.load(os.path.join(r, 'all_probabilities.npy'))
        if all_probabilities is None:
            all_probabilities = p
        else:
            if p.shape != all_probabilities.shape:
                print('Error: all_probabilities.npy do not match: ' + str(p.shape) + ' vs. ' + str(all_probabilities.shape) + ' for ' + r)
                sys.exit(1)

            if ENSEMBLE_MAX:
                all_probabilities = np.maximum(all_probabilities, p)
            else:
                all_probabilities = all_probabilities + p

    if ENSEMBLE_MAX == False:
        all_probabilities = all_probabilities / len(r)

    # load atlas images
    putil.load_atlas_images(FLAGS.data_atlas_dir)

    pre_process_params = {'zscore_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    t = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
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

    index = 0
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

            probabilities = all_probabilities[index, :, :]
            index = index + 1
            predictions = LABEL_CLASSES[probabilities.argmax(axis=1)]

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
        print('Ensemble from ' + str(RESULTS), file=summary_file)
        print('ENSEMBLE_MAX ' + str(ENSEMBLE_MAX), file=summary_file)
        stats = statistics.gather_statistics(os.path.join(result_dir, 'results.csv'))
        print('Result statistics:', file=summary_file)
        print(stats, file=summary_file)


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')
    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result-ensemble')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
