"""This module contains utility classes and functions."""

import os

import numpy as np
import SimpleITK as sitk

import mialab.classifier.decision_forest as df
import mialab.data.loading as load
import mialab.data.structure as structure
import mialab.evaluation.evaluator as eval
import mialab.evaluation.metric as metric
import mialab.evaluation.validation as valid
import mialab.filtering.filter as fltr
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.filtering.registration as fltr_reg


class FeatureExtractor:
    """Represents a feature extractor."""
    
    def __init__(self, img: structure.BrainImage):
        pass
        #


def process(id_: str, paths: dict):
    """todo(fabianbalsiger): comment
    Args:
        id_ (str): An image identifier.
        paths (str): 
    Returns:
    """

    print('-' * 5, 'Processing', id_)

    # load image
    path = paths.pop(id_, '')
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    img = structure.BrainImage(id_, path, img)

    # construct pipeline
    pipeline = fltr.FilterPipeline()
    pipeline.add_filter(fltr_prep.NormalizeZScore())
    pipeline.add_filter(fltr_reg.MultiModalRegistration())
    pipeline.set_param(fltr_reg.MultiModalRegistrationParams(T1_ATLAS_IMG), 1)

    img.images[structure.BrainImageTypes.T1] = pipeline.execute(img.images[structure.BrainImageTypes.T1])
    pipeline.set_param(fltr_reg.MultiModalRegistrationParams(T2_ATLAS_IMG), 1)
    img.images[structure.BrainImageTypes.T2] = pipeline.execute(img.images[structure.BrainImageTypes.T2])

    return img


def postprocess(id_: str, img: structure.BrainImage, segmentation, probability) -> sitk.Image:
    
    print('-' * 5, 'Post-process', id_)
    
    # construct pipeline
    pipeline = fltr.FilterPipeline()
    pipeline.add_filter(fltr_postp.DenseCRF())
    pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1],
                                                 img.images[structure.BrainImageTypes.T2],
                                                 probability), 0)
    
    return pipeline.execute(segmentation)

    
def init_evaluator(directory: str) -> eval.Evaluator:
    """Initializes an evaluator.
    Args:
        directory (str): The directory for the results file.
    Returns:
        eval.Evaluator: An evaluator.
    """
    os.makedirs(FLAGS.result_dir, exist_ok=True)  # generate result directory, if it does not exists

    evaluator = eval.Evaluator(eval.ConsoleEvaluatorWriter(5))
    evaluator.add_writer(eval.CSVEvaluatorWriter(os.path.join(directory, 'results.csv')))
    evaluator.add_label(1, "WhiteMatter")
    evaluator.metrics = [metric.DiceCoefficient()]
    return evaluator


class BrainImageFilePathGenerator(load.FilePathGenerator):

    @staticmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        if file_key == structure.BrainImageTypes.T1:
            file_name = 'T1native_biasfieldcorr_noskull'
        elif file_key == structure.BrainImageTypes.T2:
            file_name = 'T2native_biasfieldcorr_noskull'
        elif file_key == structure.BrainImageTypes.GroundTruth:
            file_name = 'labels_native'
        elif file_key == structure.BrainImageTypes.BrainMask:
            file_name = 'Brainmasknative'
        else:
            raise ValueError('Unknown key')

        return os.path.join(root_dir, file_name + file_extension)


def load_atlas(dir_path: str):
    T1_ATLAS_IMG = sitk.ReadImage(os.path.join(dir_path, 'sometext.nii.gz'))
    T2_ATLAS_IMG = sitk.ReadImage(os.path.join(dir_path, 'sometext.nii.gz'))