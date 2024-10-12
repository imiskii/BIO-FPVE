"""
@file: image.py
@brief: Image classes for easier work with images
@author: Michal Ľaš (xlasmi00)
@date: 01.10.2024
"""

from typing import Any, Self, NewType
from functools import wraps
from copy import deepcopy
import cv2 as cv
import numpy as np


ImageSet = NewType("ImageSet", None)

class Image:
    """
    Class representing an image.
    """

    def __init__(self, image_data: np.ndarray, name:str="Untitled_image") -> None:
        """
        Image constructor.

        `image_data`: data
        `name`: name of the image
        """
        self._data:np.ndarray = image_data
        self._color:int = cv.IMREAD_GRAYSCALE if len(image_data.shape) < 3 else cv.IMREAD_COLOR
        self.name:str = name


    def _add__(self, other:Self) -> Self:
        return Image(cv.add(self._data, other.GetData()), self._color, f"{self.name}_add_{other.name}")


    def __sub__(self, other:Self) -> Self:
        return Image(cv.subtract(self._data, other.GetData()), self._color, f"{self.name}_add_{other.name}")


    def apply_on_copy(method):
        """
        Wrapper for methods that modify Image object. This wraper alows them to create a deep copy and apply the method
        on the copy insted of original object. The method has to have specified parameter inplace=False to create a deepcopy.
        Inplace is set to True by default.
        """
        @wraps(method)
        def wrapper(self, *args, inplace:bool=True, **kwargs) -> Self:
            # Modify current image
            if inplace:
                method(self, *args, **kwargs)
                return self
            else:
                # Create a deep copy and apply the method to the copy
                new_img = deepcopy(self)
                method(new_img, *args, **kwargs)
                return new_img
        return wrapper
    

    def GetData(self) -> np.ndarray:
        return self._data
    

    def GetColor(self) -> int:
        return self._color


    def GetSize(self) -> tuple[int, int]:
        """
        Return height and width of the image.
        """
        return self._data.shape[0], self._data.shape[1]
    

    def GetBitDepth(self) -> np.dtypes:
        return self._data.dtype


    def IsGrayscale(self) -> bool:
        if self._color == cv.IMREAD_GRAYSCALE:
            return True
        else:
            return False


    @apply_on_copy
    def ConvertToGrayscale(self) -> Self:
        if self._color != cv.IMREAD_GRAYSCALE:
            self._data = cv.cvtColor(self._data, cv.COLOR_BGR2GRAY)
            self._color = cv.IMREAD_GRAYSCALE


    def Copy(self) -> Self:
        """
        Creates and returns a deep copy of the current Image object.
        """
        return deepcopy(self)


    @apply_on_copy
    def Resize(self, new_width, new_height) -> Self:
        """Change the image size."""
        self._data = cv.resize(self._data, (new_width, new_height))


    @apply_on_copy
    def Scale(self, scale_factor:float) -> Self:
        height, width = self.GetSize()
        self._data = cv.resize(self._data, (int(height * scale_factor), int(width * scale_factor)))


    @apply_on_copy
    def Crop(self, x, y, width, height) -> Self:
        """Crop the image on selected values."""
        self._data = self._data[y:y+height, x:x+width]


    @apply_on_copy
    def GaussianBlur(self, kernel_size:tuple[int]=(5,5), sigmaX:int=0) -> Self:
        """
        Apply GaussianBlur on the image.

        `kernel_size`: size of Gausian blur kernel (matrix size)
        `sigmaX`: amount of used filter
        
        Return new image with applied filter.
        """
        self._data = cv.GaussianBlur(self._data, kernel_size, sigmaX)
    

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