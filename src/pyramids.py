"""
@file: pyramids.py
@brief: Pyramids algorithms: Gaussian, Laplacian, and Detail
@author: Michal Ľaš (xlasmi00)
@date: 05.10.2024
"""

import cv2 as cv
from image import Image
from image_set import ImageSet


def gaussian_pyramid(image:Image, depth:int, new_set_name:str="Gaussian_Pyramid") -> ImageSet:
    """
    Count Gaussian Pyramid from given `image` with pyramid level `depth`.
    """
    gauss_pyramid = ImageSet([image], f"{new_set_name}_gaussian_pyramid")
    img = image.GetData().copy()
    for i in range(depth):
        img = cv.pyrDown(img)
        gauss_pyramid.Append(Image(img, image.GetColor(), f"{new_set_name}-{i}"))
    return gauss_pyramid


def laplacian_pyramid(gaussian_pyramid:ImageSet, new_set_name:str="Laplacian_Pyramid") -> ImageSet:
    """
    Count Laplacian Pyramid from given `gaussian_pyramid`.
    """
    lapla_pyramid = ImageSet([gaussian_pyramid[-1]], new_set_name)
    for i in range((len(gaussian_pyramid) - 1), 0, -1):
        size = gaussian_pyramid[i-1].GetSize() # Calculation of final size!
        gauss_up_one = cv.pyrUp(gaussian_pyramid[i].GetData(), dstsize=(size[1], size[0])) # the size have to be used because of sequence pyrDown and pyrUp on odd heights/widths does not create image of original size
        lapla = cv.subtract(gaussian_pyramid[i-1].GetData(), gauss_up_one)
        lapla_pyramid.Append(Image(lapla, gaussian_pyramid[i-1].GetColor(), f"{new_set_name}-{i}"))
    return lapla_pyramid 


def detail_pyramid(gaussian_pyramid:ImageSet, new_set_name:str="Detail_Pyramid") -> list[Image]:
    """
    Count Detail Pyramid from given `gaussian_pyramid`.
    """
    detail_pyramid = ImageSet([], new_set_name)
    for i, image in enumerate(gaussian_pyramid):
        blurred = image.GaussianBlur()
        detail = cv.subtract(image.GetData(), blurred.GetData())
        detail_pyramid.Append(Image(detail, image.GetColor(), f"{new_set_name}-{i}"))
    return detail_pyramid