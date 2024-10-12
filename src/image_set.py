"""
@file: image_set.py
@brief: ImageSet classe for easier work with set of images
@author: Michal Ľaš (xlasmi00)
@date: 01.10.2024
"""

import matplotlib.pyplot as plt
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


    def ConvertAllToGrayscale(self) -> None:
        for image in self._images:
            image.ConvertToGrayscale()


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


    def Show(self, max_window_width: int = 1600, max_window_height: int = 800) -> None:
        """
        Display a set of images in one window. The window can be closed by pressing any key.
        Parameters:
            `max_window_width`: the maximum width of the window with images
            `max_window_height`: the maximum height of the window with images
        """

        if self._image_count == 0:
            print(f"Image set {self.name} is empty. There is nothing to show.")
            return

        # Order images from biggest width to smallest
        images = self._images.copy()
        images.sort(key=lambda img: img.GetSize()[1], reverse=True)

        # Adjust max_window_width if the largest image is too large
        if max_window_width < images[0].GetSize()[1]:
            max_window_width = images[0].GetSize()[1]

        # Extract the heights and widths of all images
        heights = []
        widths = []
        image_matrix = []
        row = []
        total_height = images[0].GetSize()[0]
        total_width = 0
        row_width = 0
        for img in images:
            height, width = img.GetSize()
            # if image is too large
            if width > max_window_width:
                scale = max_window_width / width
                width = int(width * scale)
                height = int(height * scale)
                img.Scale(scale)
            
            heights.append(height)
            widths.append(width)
            # Create image matrix
            if row_width + width > max_window_width:
                image_matrix.append(row)
                row = [img]
                total_width = max(row_width, total_width)
                row_width = width
                total_height += height
            else:
                row.append(img)
                row_width += width

        # Add last row
        image_matrix.append(row)
        total_width = max(row_width, total_width)

        # Check if scaling is required (if total height exceeds max_window_height)
        scale_factor = 1.0
        if total_height > max_window_height:
            scale_factor = max_window_height / total_height
            total_height = int(total_height * scale_factor) + 1
            total_width = int(total_width * scale_factor) + 1

        # Create empty canvas
        stacked_image = np.zeros((total_height, total_width), dtype=np.uint16)

        # Fill the canvas with images row by row
        idx = 0
        y_offset = 0
        x_offset = 0
        last_row_max_height = 0
        for row in image_matrix:
            for img in row:
                img_height = heights[idx]
                img_width = widths[idx]

                # Resize the image based on scale factor if needed
                if scale_factor < 1.0:
                    img.Scale(scale_factor)
                    img_height = int(img_height * scale_factor)
                    img_width = int(img_width * scale_factor)

                # Handle 8-bit and 16-bit images uniformly
                if img.GetBitDepth() == np.uint8:
                    stacked_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img.GetData().astype(np.uint16) * 256
                else:
                    stacked_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img.GetData()

                idx += 1
                x_offset += img_width
                last_row_max_height = max(last_row_max_height, img_height)

            y_offset += last_row_max_height
            x_offset = 0

        # Display the final stacked image
        cv.imshow(self.name, stacked_image)
        cv.waitKey(0)


    def SlideShow(self, max_window_width: int = 1600, max_window_height: int = 800) -> None:
        """
        Display images (Grayscale only!) from the ImageSet as a slideshow.
        """
        if self._image_count == 0:
            print(f"Image set {self.name} is empty. There is nothing to show.")
            return

        # Copy and scale images
        images = self._images.copy()
        for image in images:
            img_height, img_width = image.GetSize()
            if img_height > max_window_height or img_width > max_window_width:
                scale_factor = min(max_window_height / img_height, max_window_width / img_width)
                image.Scale(scale_factor)

        # Initialize current image index
        self.__current_index = 0

        # Create a figure for the slideshow
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        # Method to update the image in the slideshow
        def resize_and_display_image():
            image = images[self.__current_index]
            depth = 65536 if image.GetBitDepth() == np.uint16 else 255
            # Display the image
            ax.clear()
            ax.imshow(image.GetData(), cmap='gray', vmin=0, vmax=depth)
            ax.set_title(f"{image.name} ({self.__current_index + 1}/{self._image_count})")
            plt.draw()

        # Key event handler to move forward or backward
        def on_key(event):
            if event.key == 'right':  # Move forward
                self.__current_index = (self.__current_index + 1) % self._image_count
            elif event.key == 'left':  # Move backward
                self.__current_index = (self.__current_index - 1) % self._image_count
            resize_and_display_image()

        # Connect the key press event
        fig.canvas.mpl_connect('key_press_event', on_key)

        # Show the first image
        resize_and_display_image()
        fig.canvas.toolbar.update()
        plt.show()




class SlideShow:

    def __init__(self, images:list) -> None:
        self._images = images
        self._image_count = len(images)
        self._current_index = 0
