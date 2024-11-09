"""
@file: fusion.py
@brief: Image fusion algorithms
@author: Tomáš Bártů (xbartu11)
@date: 7.11.2024
"""

import cv2 as cv
import numpy as np
import pywt
from sklearn.decomposition import PCA


from image import Image
from image_set import ImageSet


def process_fusion(image_sets: list[ImageSet], method: str = 'wavelet', **kwargs) -> ImageSet:
    """
    Process each group of images using the specified fusion method.

    `image_sets`: List of ImageSets to be processed.
    `method`: The fusion method to use ('wavelet', 'hdr', 'exposure', 'average', 'laplacian').

    Returns an ImageSet containing the fused images.
    """
    fused_images = []

    for image_set in image_sets:
        print(f'Processing group with prefix: {image_set.name}')

        if method == 'wavelet':
            fused_image = image_set.WaveletFusion(
                wavelet=kwargs.get('wavelet', 'sym5'),
                level=kwargs.get('level', 5),
                combine_method=kwargs.get('combine_method', 'mean')
            )
        elif method == 'exposure':
            fused_image = exposure_fusion(image_set)
        elif method == 'average':
            fused_image = average_fusion(image_set)
        elif method == 'laplacian':
            fused_image = laplacian_pyramid_fusion(image_set, max_level=9)
        elif method == 'pca':
            fused_image = pca_fusion(image_set)
        else:
            raise ValueError("Unsupported fusion method. Use 'wavelet', 'exposure', 'average', or 'laplacian'.")

        fused_images.append(fused_image)

    return ImageSet(fused_images, set_name="Fused_ImageSet")


def wavelet_fusion(images: ImageSet, wavelet: str = 'db1', level: int = None, combine_method: str = 'mean') -> Image:
    """
    Perform wavelet-based image fusion with parameterization.

    `images`: ImageSet to be fused (assumed to be uint16).
    `wavelet`: Type of wavelet to use (e.g., 'db1', 'haar', 'sym5').
    `level`: Level of decomposition for the wavelet transform.
    `combine_method`: Method to combine coefficients ('max', 'mean', 'min').

    Returns an Image object representing the fused image.
    """
    decomposed_images = [pywt.wavedec2(img.GetData(), wavelet, level=level) for img in images]

    fused_coeffs = decomposed_images[0]
    for coeffs in decomposed_images[1:]:
        if combine_method == 'max':
            fused_coeffs[0] = np.maximum(fused_coeffs[0], coeffs[0])
            fused_coeffs[1:] = [(np.maximum(c1[0], c2[0]), np.maximum(c1[1], c2[1]), np.maximum(c1[2], c2[2]))
                                for c1, c2 in zip(fused_coeffs[1:], coeffs[1:])]
        elif combine_method == 'mean':
            fused_coeffs[0] = (fused_coeffs[0] + coeffs[0]) / 2
            fused_coeffs[1:] = [((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, (c1[2] + c2[2]) / 2)
                                for c1, c2 in zip(fused_coeffs[1:], coeffs[1:])]
        elif combine_method == 'min':
            fused_coeffs[0] = np.minimum(fused_coeffs[0], coeffs[0])
            fused_coeffs[1:] = [(np.minimum(c1[0], c2[0]), np.minimum(c1[1], c2[1]), np.minimum(c1[2], c2[2]))
                                for c1, c2 in zip(fused_coeffs[1:], coeffs[1:])]
        else:
            raise ValueError("Unsupported combine method. Use 'max', 'mean', or 'min'.")

    fused_image = pywt.waverec2(fused_coeffs, wavelet)
    fused_image = Image(np.clip(fused_image, 0, 65535).astype(np.uint16), name=f'{images.name}_fusion')
    return fused_image


def average_fusion(images: ImageSet) -> Image:
    """
    Perform simple averaging image fusion.

    `images`: ImageSet containing images to be fused (assumed to be uint16).

    Returns an Image object representing the fused image.
    """
    if len(images) == 0:
        raise ValueError("ImageSet is empty. Cannot perform fusion.")

    summed_image = np.zeros_like(images[0].GetData(), dtype=np.float64)

    for img in images:
        summed_image += img.GetData().astype(np.float64)

    averaged_image = summed_image / len(images)
    averaged_image = np.clip(averaged_image, 0, 65535).astype(np.uint16)
    fused_image = Image(averaged_image, name=f'{images.name}_average_fusion')

    return fused_image


def build_laplacian_pyramid(image, levels):
    """
    Build a Laplacian pyramid for the given image.

    `image`: The input image (assumed to be grayscale or single-channel).
    `levels`: Number of levels for the pyramid.

    Returns a list representing the Laplacian pyramid.
    """
    gaussian_pyramid = [image]
    for _ in range(levels):
        image = cv.pyrDown(image)
        gaussian_pyramid.append(image)

    laplacian_pyramid = []
    for i in range(levels, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        expanded = cv.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv.subtract(gaussian_pyramid[i - 1], expanded)
        laplacian_pyramid.append(laplacian)

    return laplacian_pyramid


def laplacian_pyramid_fusion(image_set: ImageSet, max_level: int = 9) -> Image:
    """
    Perform Laplacian pyramid fusion on an ImageSet.

    `image_set`: ImageSet to be fused.
    `max_level`: Number of levels in the pyramid.

    Returns an Image object representing the fused image.
    """
    # Create Laplacian pyramids for all images in the set
    pyramids = [build_laplacian_pyramid(img.GetData(), max_level) for img in image_set]

    fused_pyramid = []
    for level in range(max_level):
        # Collect all images from the current level across pyramids
        level_images = [pyr[level] for pyr in pyramids if level < len(pyr)]
        fused_level = np.mean(level_images, axis=0)
        fused_pyramid.append(fused_level)

    # Reconstruct the final image from the fused pyramid
    fused_image = fused_pyramid[0]
    for level in range(1, max_level):
        target_shape = (fused_pyramid[level].shape[1], fused_pyramid[level].shape[0])
        fused_image = cv.pyrUp(fused_image, dstsize=target_shape)
        fused_image = cv.add(fused_image, fused_pyramid[level])

    fused_image = np.clip(fused_image, 0, 65535).astype(np.uint16)
    return Image(fused_image, name=f'{image_set.name}_laplacian_fusion')


def exposure_fusion(images: ImageSet, contrast_weight: float = 1.0, saturation_weight: float = 1.0, well_exposedness_weight: float = 1.0) -> Image:
    """
    Perform exposure fusion on a set of images.

    `images`: ImageSet containing images to be fused (assumed to be uint16).
    `contrast_weight`: Weight for the contrast component.
    `saturation_weight`: Weight for the saturation component.
    `well_exposedness_weight`: Weight for the well-exposedness component.

    Returns an Image object representing the fused image.
    """
    images_8u = [img.GetData().astype(np.uint8) for img in images]

    mertens = cv.createMergeMertens(contrast_weight, saturation_weight, well_exposedness_weight)

    fused_image = mertens.process(images_8u)

    fused_image_16u = (fused_image * 65535).astype(np.uint16)

    return Image(fused_image_16u, name=f'{images.name}_exposure_fusion')


def pca_fusion(images: ImageSet) -> Image:
    image_data = [img.GetData().astype(np.float32).flatten() for img in images]
    data_matrix = np.array(image_data)

    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(data_matrix.T).flatten()

    fused_image = principal_component.reshape(images[0].GetData().shape)
    fused_image = np.clip(fused_image, 0, 65535).astype(np.uint16)
    return Image(fused_image, name=f'{images.name}_pca_fusion')

