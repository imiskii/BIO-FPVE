#
# @file: image.py
# @description: Image and ImageSet classes for easier work with images
# @date: 01.10.2024
#

import cv2 as cv
import numpy as np


class Image:
    """
    Class representing an image.
    """

    def __init__(self, image_data: np.ndarray, name:str="Untitled image") -> None:
        self.data:np.ndarray = image_data
        self.name:str = name



    def GetSize(self) -> tuple[int, int]:
        """
        Return height and width of the image.
        """
        height, width, _ = self.data.shape
        return height, width



    def Show(self) -> None:
        """
        Display image in separate window. Window can be closed by pressing any key.
        """
        cv.imshow(self.name, self.data)
        cv.waitKey(0)



class ImageSet:
    """
    Ordered set of Image objects. Default order is by brightness.
    """

    def __init__(self, image_list:list[Image], set_name:str="Untitled image set") -> None:
        self.images:list[Image] = image_list
        self.SortByBrightness()
        self.image_count = len(self.images)
        self.name:str = set_name


    def SortByBrightness(self) -> None:
        """
        Sort images by their brightness level in ascending order.
        """
        self.images.sort(key=lambda img: np.mean(img.data), reverse=True)


    def Show(self, max_window_width:int=1600) -> None:
        """
        Display set of images in separate window. Window can be closed by pressing any key.
        Parameters:
            `max_window_width`: the maximum width of window with images
        """

        if not self.images:
            print(f"Image set {self.name} is empty. There is nothing to show.")
            return
        
        # Find maximal width and height of images
        heights = [image.data.shape[0] for image in self.images]
        widths = [image.data.shape[1] for image in self.images]

        max_height = max(heights)
        max_width = max(widths)

        # Count number of rows
        num_rows = 1
        while sum(widths[:(self.image_count // num_rows)]) > max_window_width:
            num_rows += 1
        images_per_row = (self.image_count + num_rows - 1) // num_rows

        # Create empty canvas
        stacked_image = np.zeros((max_height * num_rows, max_width * images_per_row, 3), dtype=np.uint8)

        # Fill canvas
        for idx, img in enumerate(self.images):
            row = idx // images_per_row
            col = idx % images_per_row

            y_offset = row * max_height
            x_offset = col * max_width

            stacked_image[y_offset:y_offset + img.data.shape[0], x_offset:x_offset + img.data.shape[1]] = img.data

        cv.imshow(self.name, stacked_image)
        cv.waitKey(0)