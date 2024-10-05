"""
@file: image.py
@brief: Image classes for easier work with images
@author: Michal Ľaš (xlasmi00)
@date: 01.10.2024
"""

from typing import Any, Self, NewType
from copy import deepcopy
import cv2 as cv
import numpy as np


ImageSet = NewType("ImageSet", None)

class Image:
    """
    Class representing an image.
    """

    IMG_GRAYSCLACE:int = 0
    IMG_COLOR:int = 1

    def __init__(self, image_data: np.ndarray, color:int, name:str="Untitled_image") -> None:
        """
        Image constructor.

        `image_data`: data
        `color`: color base of image - ` IMG_GRAYSCLACE`/`IMG_COLOR`
        `name`: name of the image
        """
        self._data:np.ndarray = image_data
        self._color:int = color
        self.name:str = name


    def GetData(self) -> np.ndarray:
        return self._data
    

    def GetColor(self) -> int:
        return self._color


    def GetSize(self) -> tuple[int, int]:
        """
        Return height and width of the image.
        """
        if self._color == self.IMG_GRAYSCLACE:
            height, width = self._data.shape
        else:
            height, width, _ = self._data.shape
        return height, width


    def Copy(self) -> Self:
        """
        Creates and returns a deep copy of the current Image object.
        """
        return deepcopy(self)


    def Resize(self, new_width, new_height) -> None:
        """Change the image size."""
        self._data = cv.resize(self._data, (new_width, new_height))


    def Crop(self, x, y, width, height):
        """Crop the image on selected values."""
        self._data = self._data[y:y+height, x:x+width]


    def GaussianBlur(self, kernel_size:tuple[int]=(5,5), sigmaX:int=0) -> None:
        """
        Apply GaussianBlur on the image.

        `kernel_size`: size of Gausian blur kernel (matrix size)
        `sigmaX`: amount of used filter
        
        Return new image with applied filter.
        """
        self._data = cv.GaussianBlur(self._data, kernel_size, 0)
    

    def GaussianPyramid(self, depth:int) -> ImageSet:
        """
        Create Gaussian Pyramid of this image.
        
        `depth`: number of Gaussian Pyramid levels
        """
        from pyramids import gaussian_pyramid
        return gaussian_pyramid(self, depth, f"{self.name}_GaussPyr")


    def LaplacianPyramid(self, depth:int) -> ImageSet:
        """
        Create Laplacian Pyramid of this image.

        `depth`: number of Laplacian Pyramid levels
        """
        from pyramids import laplacian_pyramid
        return laplacian_pyramid(self.GaussianPyramid(depth), f"{self.name}_LaplacPyr")
    

    def DetailPyramid(self, depth:int) -> list[Self]:
        """
        Create Detail Pyramid of this image.

        `depth`: number of Detail Pyramid levels
        """
        from pyramids import detail_pyramid
        return detail_pyramid(self.GaussianPyramid(depth), f"{self.name}_DetailPyr")


    def Show(self) -> None:
        """
        Display image in separate window. Window can be closed by pressing any key.
        """
        cv.imshow(self.name, self._data)
        cv.waitKey(0)