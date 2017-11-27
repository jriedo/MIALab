import SimpleITK as sitk
import os
import sys
import numpy as np
import paraview.simple as simple



if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)


    prediction=sitk.ReadImage('mia-result/DF_trees_160_nodes_3000/188347_SEG.mha')
    prediction=sitk.GetArrayFromImage(prediction)

    ground_truth = sitk.ReadImage('../data/test/188347/labels_mniatlas.nii.gz')
    ground_truth = sitk.GetArrayFromImage(ground_truth)

    error_mask=np.zeros_like(ground_truth)

    # 0 background correct labeled
    # error_mask[np.where(np.logical_and(ground_truth == 0, prediction == 0))]=0
    # 0 background wrong labeled
    error_mask[np.where(np.logical_and(ground_truth == 0, prediction != 0))] = 1

    # 1 white matter correct labeled
    # error_mask[np.where(np.logical_and(ground_truth == 1, prediction == 1))] = 2
    # 1 white matter wrong labeled
    error_mask[np.where(np.logical_and(ground_truth == 1, prediction != 1))] = 3

    # 2 grey matter correct labeled
    # error_mask[np.where(np.logical_and(ground_truth == 2, prediction == 2))] = 4
    # 2 grey matter wrong labeled
    error_mask[np.where(np.logical_and(ground_truth == 2, prediction != 2))] = 5

    # 3 ventricles correct labeled
    # error_mask[np.where(np.logical_and(ground_truth == 3, prediction == 3))] = 6
    # 3 ventricles wrong labeled
    error_mask[np.where(np.logical_and(ground_truth == 3, prediction != 3))] = 7

    error_mask=sitk.GetImageFromArray(error_mask)
    sitk.WriteImage(error_mask, '../data/test/188347/error_mask_SEG.mha')