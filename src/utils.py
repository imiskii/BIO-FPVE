"""
@file: utils.py
@brief: Basic utilities for processing images
@author: Michal Ľaš (xlasmi00)
@date: 01.10.2024
"""

from os import listdir
from os.path import isfile, join, splitext
import cv2 as cv
from image import Image
from image_set import ImageSet


def load_image_grayscale(image_path:str) -> Image:
    """
    Load one image and make it greyscale. Return Image object.

    `image_path`: absolute/relative path to the image
    
    >>> image:Image = load_image("merged_vein.png")
    """
    img = cv.imread(cv.samples.findFile(image_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception(f"Could not read the image: {image_path}")

    return Image(img, Image.IMG_GRAYSCLACE, image_path.rpartition('/')[-1])


def load_image_color(image_path:str):
    """
    Load one image in its color. Return Image object.

    `image_path`: absolute/relative path to the image
    
    >>> image:Image = load_image("merged_vein.png")
    """
    img = cv.imread(cv.samples.findFile(image_path))
    if img is None:
        raise Exception(f"Could not read the image: {image_path}")

    return Image(img, Image.IMG_COLOR, image_path.rpartition('/')[-1])


def load_image_set(image_paths_list:str, images_color:str= 'GRAYSCALE') -> ImageSet:
    """
    Load set of images from given directory. It treats all files in specified directory as images! Returns ImageSet object.

    `image_paths_list`: absolute/relative path to a directory with images
    `images_color`: Color in which images will be loaded can be GRAYSCALE/COLOR

    >>> imgs = load_image_set(['data/example_splited/column-1.png', 'data/example_splited/column-2.png', 'data/example_splited/column-3.png'])
    >>> imgs = load_image_set(find_images_in_directory("data/example_splited/"))
    """

    match images_color:
        case 'GRAYSCALE':
            return ImageSet([load_image_grayscale(image) for image in image_paths_list])
        case 'COLOR':
            return ImageSet([load_image_color(image) for image in image_paths_list])
        case _:
            raise Exception(f"Unknow images_color argument in function load_image_set: {images_color}")




def find_images_in_directory(directory:str) -> list[str]:
    """
    Returns list of image files (.png, .jpg, .jpeg) paths in given directory.
    >>> imgs = find_images_in_directory("data/example_splited/")
    """
    result = []
    for file in listdir(directory):
        path = join(directory, file)
        if isfile(path) and splitext(path)[1] in ['.png', '.jpg', '.jpeg']:
            result.append(path)
    return result


