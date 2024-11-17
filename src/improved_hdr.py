"""
@file: improved_hdr.py
@brief: Improved High Dynamic Range (HDR) image processing
@author: Michal Ľaš (xlasmi00)
@date: 05.10.2024


This algorithm was inspired by the article https://www.sciencedirect.com/science/article/pii/S1350449521000700:

@author: Hang Gao and Qian Chen and Chengwei Liu and Guohua Gu
@title: High dynamic range infrared image acquisition based on an improved multi-exposure fusion algorithm
@journal: Infrared Physics & Technology
@volume: 115
@pages: 103698
@year: 2021
@issn: 1350-4495
@doi: https://doi.org/10.1016/j.infrared.2021.103698


"""

from image import Image
from image_set import ImageSet
from pyramids import gaussian_pyramid, laplacian_pyramid, detail_pyramid

import numpy as np
import cv2 as cv


def calc_contrast(gauss_layer_0:Image, detail_layer_0:Image) -> np.ndarray:
    """
    Calculates the contrast map based on Gaussian and detail layers.

    Parameters:
        `gauss_layer_0` (Image): The base Gaussian layer of the image.
        `detail_layer_0` (Image): The corresponding detail layer of the image.

    Returns:
        np.ndarray: The contrast map as a numpy array.
    """
    return np.abs(detail_layer_0.GetData() / (gauss_layer_0.GaussianBlur().GetData() + 1e-5))


def calc_well_exposedness(image:Image, image_data_mean:float, sigma_g=0.5, sigma_l=0.2) -> np.ndarray:
    """
    Calculates the well-exposedness map for an image using a global and local Gaussian distribution.

    Parameters:
        `image` (Image): Input image.
        `image_data_mean` (float): Mean value of the image data.
        `sigma_g` (float, optional): Global exposure standard deviation. Default is 0.5.
        `sigma_l` (float, optional): Local exposure standard deviation. Default is 0.2.

    Returns:
        np.ndarray: The well-exposedness map as a numpy array.
    """
    image_data = cv.normalize(image.GetData(), None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    global_wex = -(((image_data_mean - 0.5) ** 2) / (2 * sigma_g ** 2))
    
    def calc_pixel_we(pixel_value:float):
        local_wex = ((pixel_value - 0.5) ** 2) / (2 * sigma_l ** 2)
        return  np.exp(global_wex - local_wex)
    
    calc_pixel_we_vectorized = np.vectorize(calc_pixel_we)
    return calc_pixel_we_vectorized(image_data)


def calc_brightness(image:Image, image_data_mean:float) -> np.ndarray:
    """
    Computes the brightness weight map for an image.

    Parameters:
        `image` (Image): Input image.
        `image_data_mean` (float): Mean value of the image data.

    Returns:
        np.ndarray: The brightness weight map as a numpy array.
    """
    brightness_weight = image.GetData() / image_data_mean
    return brightness_weight / brightness_weight.max()


def compute_weight_map(image:Image, gauss_layer_0:Image, detail_layer_0:Image, image_data_mean:float, alpha:int=1, beta:int=1, gamma:int=1) -> np.ndarray:
    """
    Computes the combined weight map based on contrast, well-exposedness, and brightness maps.

    Parameters:
        `image` (Image): Input image.
        `gauss_layer_0` (Image): The base Gaussian layer.
        `detail_layer_0` (Image): The corresponding detail layer.
        `image_data_mean` (float): Mean value of the image data.
        `alpha` (int, optional): Weight exponent for contrast. Default is 1.
        `beta` (int, optional): Weight exponent for well-exposedness. Default is 1.
        `gamma` (int, optional): Weight exponent for brightness. Default is 1.

    Returns:
        np.ndarray: The combined weight map.
    """
    contrast = calc_contrast(gauss_layer_0, detail_layer_0)
    well_exposedness = calc_well_exposedness(image, image_data_mean)
    brightness = calc_brightness(image, image_data_mean)

    weight_map = (contrast ** alpha) * (well_exposedness ** beta) * (brightness ** gamma)
    return weight_map


def weight_maps_sum(weight_maps:list[np.ndarray]) -> float:
    """
    Computes the sum of multiple weight maps.

    Parameters:
        `weight_maps` (list[np.ndarray]): List of weight maps.

    Returns:
        float: The sum of weight maps.
    """
    weight_map_sum = weight_maps[0]
    for w_map in weight_maps[1:]:
        weight_map_sum += w_map
    weight_map_sum += 1e-5 # Add small constant to avoid zero division
    return weight_map_sum


def normalize_weight_maps(weight_maps:list[np.ndarray]) -> list[Image]:
    """
    Normalizes a list of weight maps by their sum.

    Parameters:
        `weight_maps` (list[np.ndarray]): List of weight maps.

    Returns:
        list[Image]: List of normalized weight maps as Image objects.
    """
    weight_map_sum = weight_maps_sum(weight_maps)
    result = []
    for w_map in weight_maps:
        result.append(Image(w_map / weight_map_sum))
    return result


def weight_pyrimid_update(weight_pyramid:ImageSet, laplacian_pyramid:ImageSet, alpha:float=0.2) -> ImageSet:
    """
    Updates the weight pyramid using the laplacian pyramid and a scaling factor.

    Parameters:
        `weight_pyramid` (ImageSet): The weight pyramid.
        `laplacian_pyramid` (ImageSet): The laplacian pyramid.
        `alpha` (float, optional): Scaling factor for update. Default is 0.2.

    Returns:
        ImageSet: The updated weight pyramid.
    """
    for i in range(len(weight_pyramid) - 3, len(weight_pyramid)):
        update = weight_pyramid[i].GetData() + (alpha * np.abs(laplacian_pyramid[i].GetData()))
        weight_pyramid[i] = Image(update)
    return weight_pyramid


def weight_fusion(weight_pyramid:list[ImageSet]) -> ImageSet:
    """
    Fuses multiple weight pyramids into a single pyramid.

    Parameters:
        `weight_pyramid` (list[ImageSet]): List of weight pyramids.

    Returns:
        ImageSet: The fused weight pyramid.
    """
    weight_sum = ImageSet([])
    for level in range(len(weight_pyramid[0])):
        level_sum:Image = weight_pyramid[0][level]
        for img_pyr in weight_pyramid[1:]:
            level_sum += img_pyr[level]
        weight_sum.Append(level_sum)
    return weight_sum


def fuse_pyramids(pyramid:list[ImageSet], weight_pyramid:list[ImageSet], fused_weights:ImageSet) -> ImageSet:
    """
    Fuses multiple pyramids using their respective weight pyramids.

    Parameters:
        `pyramid` (list[ImageSet]): List of image pyramids.
        `weight_pyramid` (list[ImageSet]): List of weight pyramids.
        `fused_weights` (ImageSet): Fused weight pyramid.

    Returns:
        ImageSet: The fused image pyramid.
    """
    pyr_sum = ImageSet([])
    for level in range(len(pyramid[0])):
        level_sum:Image = pyramid[0][level] * weight_pyramid[0][level]
        for img_pyr, img_weight in zip(pyramid[1:], weight_pyramid[1:]):
            level_sum += (img_pyr[level] * img_weight[level])
        pyr_sum.Append(level_sum / fused_weights[level])
    return pyr_sum


def laplacian_detail_compensation(fused_laplacian:ImageSet, fused_detail:ImageSet, beta:float=0.1) -> ImageSet:
    """
    Compensates for details in the laplacian pyramid fusion.

    Parameters:
        `fused_laplacian` (ImageSet): Fused laplacian pyramid.
        `fused_detail` (ImageSet): Fused detail pyramid.
        `beta` (float, optional): Compensation factor. Default is 0.1.

    Returns:
        ImageSet: The compensated laplacian pyramid.
    """
    result = ImageSet([])
    for lapla, detail in zip(fused_laplacian, fused_detail):
        result.Append(Image(lapla.GetData() + beta * detail.GetData()))
    return result


def make_hdr_pyramid(final_laplacian:ImageSet, laplacian_pyramid:ImageSet) -> ImageSet:
    """
    Constructs an HDR pyramid by upscaling and blending laplacian pyramids.

    Parameters:
        `final_laplacian` (ImageSet): Final compensated laplacian pyramid.
        `laplacian_pyramid` (ImageSet): Initial laplacian pyramid.

    Returns:
        ImageSet: The HDR pyramid.
    """
    hdr_pyramid = ImageSet([])
    for f_lapla, lapla in zip(final_laplacian[1:], laplacian_pyramid[:-1]):
        size = lapla.GetSize()
        up_level = 4 * cv.pyrUp(f_lapla.GetData(), dstsize=(size[1], size[0]))
        hdr_pyramid.Append(Image(up_level + lapla.GetData()))
    hdr_pyramid.Append(laplacian_pyramid[-1])
    return hdr_pyramid


def improved_hdr(images:ImageSet, depth:int, mask:None|Image=None) -> ImageSet:
    """
    Implements the HDR imaging pipeline including weight computation, pyramid fusion, and detail enhancement.

    Parameters:
        `images` (ImageSet): Set of input images.
        `depth` (int): Depth of the pyramids.
        `mask` (None | Image, optional): Mask for selecting regions. Default is None.

    Returns:
        ImageSet: The final HDR image pyramid.
    """
    images_data = {}

    for img in images:
        data = {}

        # Normalize images
        img.Normalize()

        # Calculate mean
        if mask is not None:
            mean = np.mean(img.GetData()[mask.GetData() == 1])
        else:
            mean = np.mean(img.GetData())

        data['mean'] = mean

        # Calculate pyramid
        gaussian_pyr = gaussian_pyramid(img, depth)
        laplacian_pyr = laplacian_pyramid(gaussian_pyr)
        detail_pyr = detail_pyramid(gaussian_pyr)

        data['gauss'] = gaussian_pyr
        data['lapla'] = laplacian_pyr
        data['detail'] = detail_pyr

        # Calculate weight map
        data['w_map'] = compute_weight_map(img, gaussian_pyr[0], detail_pyr[0], mean)

        images_data[img.name] = data


    # Normalize weights
    weights = normalize_weight_maps([data['w_map'] for data in images_data.values()])
    for weight_map, item in zip(weights, images_data.values()):
        w_pyr = gaussian_pyramid(weight_map, depth)
        item['w_pyr'] = weight_pyrimid_update(w_pyr, item['lapla'])

    # Fusion
    fused_weights = weight_fusion([data['w_pyr'] for data in images_data.values()])
    fused_lapla = fuse_pyramids([data['lapla'] for data in images_data.values()], [data['w_pyr'] for data in images_data.values()], fused_weights)
    fused_detail = fuse_pyramids([data['detail'] for data in images_data.values()], [data['w_pyr'] for data in images_data.values()], fused_weights)

    final_lapla = laplacian_detail_compensation(fused_lapla, fused_detail)

    hdr_pyramid = make_hdr_pyramid(final_lapla, fused_lapla)
    return hdr_pyramid


# END OF FILE