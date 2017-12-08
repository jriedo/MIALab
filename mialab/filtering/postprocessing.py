"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import numpy as np
import pydensecrf.densecrf as crf
import pydensecrf.utils as crf_util
import SimpleITK as sitk

import mialab.filtering.filter as fltr


class DenseCRFParams(fltr.IFilterParams):
    """Dense CRF parameters."""
    def __init__(self, img_t1: sitk.Image, img_t2: sitk.Image, img_proba: sitk.Image):
        """Initializes a new instance of the DenseCRFParams
        
        Args:
            img_t1 (sitk.Image): The T1 image.
            img_t2 (sitk.Image): The T2 image.
            img_proba (sitk.Image): The posterior probability image.
        """
        self.img_t1 = img_t1
        self.img_t2 = img_t2
        self.img_proba = img_proba


class DenseCRF(fltr.IFilter):
    """A dense conditional random field (dCRF).

    Implements the work of Krähenbühl and Koltun, Efficient Inference in Fully Connected CRFs
    with Gaussian Edge Potentials, 2012. The dCRF code is taken from https://github.com/lucasb-eyer/pydensecrf.
    """

    def __init__(self):
        """Initializes a new instance of the DenseCRF class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: DenseCRFParams=None) -> sitk.Image:
        """Executes the dCRF regularization.

        Args:
            image (sitk.Image): The image (unused).
            params (IFilterParams): The parameters.

        Returns:
            sitk.Image: The filtered image.
        """

        if params is None:
            raise ValueError('Parameters are required')

        img_t2 = sitk.GetArrayFromImage(params.img_t1)
        img_ir = sitk.GetArrayFromImage(params.img_t2)
        img_proba = sitk.GetArrayFromImage(params.img_proba)

        # some variables
        x = img_proba.shape[2]
        y = img_proba.shape[1]
        z = img_proba.shape[0]
        no_labels = img_proba.shape[3]

        img_proba = np.rollaxis(img_proba, 3, 0)

        d = crf.DenseCRF(x * y * z, no_labels)  # width, height, nlabels
        U = crf_util.unary_from_softmax(img_proba)
        d.setUnaryEnergy(U)

        stack = np.stack([img_t2, img_ir], axis=3)

        # Create the pairwise bilateral term from the above images.
        # The two `s{dims,chan}` parameters are model hyper-parameters defining
        # the strength of the location and image content bilaterals, respectively.

        # higher weight equals stronger
        pairwise_energy = crf_util.create_pairwise_bilateral(sdims=(1, 1, 1), schan=(1, 1), img=stack, chdim=3)

        # `compat` (Compatibility) is the "strength" of this potential.
        compat = 10
        # compat = np.array([1, 1], np.float32)
        # weight --> lower equals stronger
        # compat = np.array([[0, 10], [10, 1]], np.float32)

        d.addPairwiseEnergy(pairwise_energy, compat=compat,
                            kernel=crf.DIAG_KERNEL,
                            normalization=crf.NORMALIZE_SYMMETRIC)

        # add location only
        # pairwise_gaussian = crf_util.create_pairwise_gaussian(sdims=(.5,.5,.5), shape=(x, y, z))
        #
        # d.addPairwiseEnergy(pairwise_gaussian, compat=.3,
        #                     kernel=dcrf.DIAG_KERNEL,
        #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

        # compatibility, kernel and normalization
        Q_unary = d.inference(10)
        # Q_unary, tmp1, tmp2 = d.startInference()
        #
        # for _ in range(10):
        #     d.stepInference(Q_unary, tmp1, tmp2)
        #     print(d.klDivergence(Q_unary) / (z* y*x))
        # kl2 = d.klDivergence(Q_unary) / (z* y*x)

        # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
        map_soln_unary = np.argmax(Q_unary, axis=0)
        map_soln_unary = map_soln_unary.reshape((z, y, x))

        map_soln_unary = map_soln_unary.astype(np.uint8)  # convert to uint8 from int64
        # Saving int64 with SimpleITK corrupts the file for Windows, i.e. opening it raises an ITK error:
        #  Unknown component type error: 0

        img_out = sitk.GetImageFromArray(map_soln_unary)
        img_out.CopyInformation(params.img_t1)
        return img_out


class LargestNConnectedComponents(fltr.IFilter):
    """Represents a largest N connected components filter.

    Extracts the largest N connected components from a label image.
    By default the N components will all have the value 1 in the output image.
    Use the `consecutive_component_labels` option such that the largest has value 1,
    the second largest has value 2, etc. Background is always assumed to be 0.
    """

    def __init__(self, number_of_components: int = 1, consecutive_component_labels: bool = False):
        """Initializes a new instance of the LargestNComponents class.

        Args:
            number_of_components (int): The number of largest components to extract.
            consecutive_component_labels (bool): The largest component has value 1, the second largest has value 2, ect.
                if set to True; otherwise, all components will have value 1.
        """
        super().__init__()

        if not number_of_components >= 1:
            raise ValueError("number_of_components must be larger or equal to 1")

        self.number_of_components = number_of_components
        self.consecutive_component_labels = consecutive_component_labels

    def execute(self, image: sitk.Image, params: fltr.IFilterParams = None) -> sitk.Image:
        """Executes the largest N connected components filter on an image.

        Args:
            image (sitk.Image): The image.
            params (IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The filtered image.
        """

        image = sitk.ConnectedComponent(image)
        image = sitk.RelabelComponent(image)

        if self.consecutive_component_labels:
            return sitk.Threshold(image, lower=1, upper=self.number_of_components, outsideValue=0)
        else:
            return sitk.BinaryThreshold(image, lowerThreshold=1, upperThreshold=self.number_of_components,
                                        insideValue=1, outsideValue=0)

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """

        return 'LargestNConnectedComponents:\n' \
               ' number_of_components:         {self.number_of_components}\n' \
               ' consecutive_component_labels: {self.consecutive_component_labels}\n' \
            .format(self=self)
