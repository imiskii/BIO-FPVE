#
# @file: utils.py
# @description: Basic utilities for processing images 
# @date: 01.10.2024
#


from os import listdir
from os.path import isfile, join, splitext
import cv2 as cv
from image import Image, ImageSet


def load_image(image_path:str) -> Image:
    """
    Load one image. Return Image object.
    >>> image:Image = load_image("merged_vein.png")
    """
    img = cv.imread(cv.samples.findFile(image_path))
    if img is None:
        raise Exception(f"Could not read the image: {image_path}")

    return Image(img, image_path.rpartition('/')[-1])



def load_image_set(image_paths_list:str) -> ImageSet:
    """
    Load set of images from given directory. It treats all files in specified directory as images! Returns ImageSet object.
    >>> imgs = load_image_set(['data/example_splited/column-1.png', 'data/example_splited/column-2.png', 'data/example_splited/column-3.png'])
    >>> imgs = load_image_set(find_images_in_directory("data/example_splited/"))
    """
    return ImageSet([load_image(image) for image in image_paths_list])



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


