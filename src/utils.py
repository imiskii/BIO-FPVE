"""
@file: utils.py
@brief: Basic utilities for processing images
@author: Michal Ľaš (xlasmi00), Tomáš Bártů (xbartu11)
@date: 01.10.2024
"""
import os
import re
from collections import defaultdict
from os import listdir
from os.path import isfile, join, splitext
import cv2 as cv
from image import Image
from image_set import ImageSet


SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def load_image(image_path:str) -> Image:
    """
    Load one image and make it greyscale. Return Image object.

    `image_path`: absolute/relative path to the image
    
    >>> image:Image = load_image("merged_vein.png")
    """
    file = splitext(image_path)
    if file[1] not in SUPPORTED_EXTENSIONS:
        raise Exception(f"File {file[0]} has unsupported extension {file[1]}!")

    img = cv.imread(cv.samples.findFile(image_path), cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read the image: {image_path}")
    
    return Image(img, image_path.rpartition('/')[-1])


def load_image_set(image_paths:list|str) -> ImageSet:
    """
    Load set of images from given directory. It treats all files in specified directory as images! Returns ImageSet object.

    `image_paths_list`: absolute/relative path to a directory with images
    `images_color`: Color in which images will be loaded can be GRAYSCALE/COLOR

    >>> imgs = load_image_set(['data/example_splited/column-1.png', 'data/example_splited/column-2.png', 'data/example_splited/column-3.png'])
    >>> imgs = load_image_set(find_images_in_directory("data/example_splited/"))
    """
    if type(image_paths) is list:
        return ImageSet([load_image(image) for image in image_paths])
    elif type(image_paths) is str:
        return ImageSet([load_image(image) for image in find_images_in_directory(image_paths)])
    else:
        raise TypeError(f"load_image_set: parameter image_paths is neither list or string. It is {type(image_paths)} type.")


def find_images_in_directory(directory:str) -> list[str]:
    """
    Returns list of image files (.png, .jpg, .jpeg) paths in given directory.
    >>> imgs = find_images_in_directory("data/example_splited/")
    """
    result = []
    for file in listdir(directory):
        path = join(directory, file)
        if isfile(path) and splitext(path)[1] in SUPPORTED_EXTENSIONS:
            result.append(path)
    return result


def get_exposure_groups(folder_path: str):
    """
    Group image files in a folder based on their exposure settings.

    `folder_path`: Path to the folder containing image files.

    Returns a dictionary where the key is the prefix of the image set and the value is a list of file paths.
    """
    image_groups = defaultdict(list)

    # Iterating through each file in the specified folder
    for filename in sorted(os.listdir(folder_path)):
        if any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            # Match the prefix and optional '_bg' or exposure suffix
            match = re.match(r'^(.*?)(_bg)?(_\d+)?_', filename)
            if match:
                # Create the prefix by concatenating the matched groups
                prefix = match.group(1) + (match.group(2) if match.group(2) else "")
                image_groups[prefix].append(os.path.join(folder_path, filename))

    return image_groups


def load_exposure_groups(folder_path: str) -> list[ImageSet]:
    """
    Load image groups from a folder and create ImageSet objects for each group.

    `folder_path`: Path to the folder containing image files.

    Returns a list of ImageSet objects, each representing a group of images.
    """
    groups = get_exposure_groups(folder_path)
    image_sets = []

    # Iterate over each group of images
    for prefix, files in groups.items():
        images = []

        # Load each image in the group and append it to the list
        for file in files:
            img = load_image(file)
            images.append(img)

        # Create an ImageSet object with the loaded images and add it to the list
        image_set = ImageSet(images, set_name=prefix)
        image_sets.append(image_set)

    return image_sets
