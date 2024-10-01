
import cv2 as cv
import numpy as np


class Image:
    """
    Class representing an image.
    """

    def __init__(self, image_data: np.ndarray, name:str="Untitled image") -> None:
        self.img:np.ndarray = image_data
        self.name:str = name


    def show(self) -> None:
        """
        Display image in separate window. Window can be closed by pressing any key.
        """
        cv.imshow(self.name, self.img)
        cv.waitKey(0)



class ImageSet:
    """
    Ordered set of Image objects.

    TODO: Add max width/height to window in show() method
    """

    def __init__(self, image_list:list[Image], set_name:str="Untitled image set") -> None:
        self.images:list[Image] = image_list
        self.name:str = set_name


    def show(self) -> None:
        """
        Display set of images in separate window. Window can be closed by pressing any key.
        """
        horizontal = np.concatenate([image.img for image in self.images], axis=1)     
        cv.imshow(self.name, horizontal)
        cv.waitKey(0)