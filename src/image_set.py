"""
@file: image_set.py
@brief: ImageSet classe for easier work with set of images
@author: Michal Ľaš (xlasmi00)
@date: 01.10.2024
"""


from typing import Any, Self
import cv2 as cv
import numpy as np
from image import Image

class ImageSet:
    """
    Set of Image objects. Uniques of Image objects is assured by image names.
    """

    def __init__(self, image_list:list[Image], set_name:str="Untitled_image_set") -> None:
        self._image_names:set[str] = set()
        self._images:list[Image] = list()
        for image in image_list:
            if image.name not in self._image_names:
                self._image_names.add(image.name)
                self._images.append(image)

        self._image_count = len(self._images)
        self.name:str = set_name


    def __getitem__(self, key:int) -> Image:
        return self._images[key]


    def __len__(self) -> int:
        return self._image_count


    def Append(self, image:Image) -> None:
        """
        Append image to the set. This may break the set ordering!
        """
        if image.name not in self._image_names:
            self._image_names.add(image.name)
            self._images.append(image)
            self._image_count += 1


    def Remove(self, index:int) -> None:
        """
        Remove image on position `key` from the set. If the position is indexed out of the set do nothing.
        """
        if 0 <= index < len(self._images):
            self._images.pop(index)
        else:
            raise IndexError(f"function: ImageSet.Remove(): index {index} is out of image set!")


    def SortByBrightness(self, reverse:bool=True) -> None:
        """
        Sort images by their brightness `reverse`=True ascending order, `reverse`=False descending order.
        """
        self._images.sort(key=lambda img: np.mean(img._data), reverse=reverse)


    def SortByWidth(self, reverse:bool=True) -> None:
        """
        Sort images by their width `reverse`=True descending order, `reverse`=False ascending order.
        """
        self._images.sort(key=lambda img: img.GetSize()[1], reverse=reverse)


    def LaplacianPyramid(self) -> Self:
        from pyramids import laplacian_pyramid
        return laplacian_pyramid(self, f"{self.name}_LaplacPyr")


    def Show(self, max_window_width: int = 1600) -> None:
        """
        Display a set of images in one window. The window can be closed by pressing any key.
        Parameters:
            `max_window_width`: the maximum width of the window with images. This walue may be overwritten if the largest image has greater width than `max_window_width`
        """

        if not self._images:
            print(f"Image set {self.name} is empty. There is nothing to show.")
            return

        # Order images from biggest width to smallest
        images = self._images.copy()
        images.sort(key=lambda img: img.GetSize()[1], reverse=True)

        # Adjust max_window_width if the largest image is too large
        if max_window_width < images[0].GetSize()[1]:
            max_window_width = images[0].GetSize()[1]

        # Extract the heights and widths of all images
        heights = [image.GetSize()[0] for image in images]
        widths = [image.GetSize()[1] for image in images]

        # Determine how many rows we need
        num_rows = 1
        while sum(widths[:(self._image_count // num_rows)]) > max_window_width:
            num_rows += 1
        images_per_row = (self._image_count + num_rows - 1) // num_rows

        # Prepare the grid to calculate total canvas size dynamically
        row_widths = []
        row_heights = []
        current_row_width = 0
        current_row_height = 0
        total_height = 0

        for idx, img in enumerate(images):
            img_height = heights[idx]
            img_width = widths[idx]
            
            if (current_row_width + img_width) > max_window_width:
                # Finish the current row, start a new row
                row_heights.append(current_row_height)
                row_widths.append(current_row_width)
                total_height += current_row_height
                current_row_width = 0
                current_row_height = 0
            
            current_row_width += img_width
            current_row_height = max(current_row_height, img_height)
            
        # Add the last row
        row_heights.append(current_row_height)
        row_widths.append(current_row_width)
        total_height += current_row_height

        # Final canvas dimensions
        canvas_width = max(row_widths)
        canvas_height = total_height

        # Create empty canvas
        stacked_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Fill the canvas with images
        y_offset = 0
        x_offset = 0
        current_row_height = 0

        for idx, img in enumerate(images):
            img_height = heights[idx]
            img_width = widths[idx]
            
            if (x_offset + img_width) > max_window_width:
                # Move to the next row
                y_offset += current_row_height
                x_offset = 0
                current_row_height = row_heights[len(row_heights) - num_rows]
            
            # Dynamically place the image based on its actual size
            resized_image = cv.resize(img.GetData(), (img_width, img_height)) # DO NOT USE img.Resize(), it needs to be a copy.
            if img.GetColor() == Image.IMG_GRAYSCLACE:
                stacked_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)
            else:
                stacked_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = resized_image
                
            x_offset += img_width
            current_row_height = max(current_row_height, img_height)

        # Display the final stacked image
        cv.imshow(self.name, stacked_image)
        cv.waitKey(0)
