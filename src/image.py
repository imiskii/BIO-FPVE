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
import pywt
from skimage.filters import gabor_kernel
from skimage.restoration import denoise_wavelet
from scipy.ndimage import convolve
from histogram_eq_16bit import histogram_equalization_16_bit



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


def get_kernel(kernel_shape: str, kernel_size: int, bit_depth: np.dtype) -> np.ndarray:
    """
    Helper function to create the kernel based on shape and size.
    """
    match kernel_shape:
        case "rectangular":
            return np.ones((kernel_size, kernel_size), bit_depth)
        case "eliptical":
            return cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        case "cross":
            return cv.getStructuringElement(cv.MORPH_CROSS, (kernel_size, kernel_size))
        case _:
            raise ValueError(f"Unsupported kernel_shape: {kernel_shape}")



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


    # Methods overloading


    def _add__(self, other:Self) -> Self:
        return Image(cv.add(self._data, other.GetData()), self._color, f"{self.name}_add_{other.name}")


    def __sub__(self, other:Self) -> Self:
        return Image(cv.subtract(self._data, other.GetData()), self._color, f"{self.name}_add_{other.name}")
    

    # Getters


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


    # Special


    def Copy(self) -> Self:
        """
        Creates and returns a deep copy of the current Image object.
        """
        return deepcopy(self)


    def Show(self) -> None:
        """
        Display image in separate window. Window can be closed by pressing any key.
        """
        cv.imshow(self.name, self._data)
        cv.waitKey(0)


    ############################################################
    ##################### IMAGE OPERATIONS #####################
    ############################################################

    @apply_on_copy
    def ConvertToGrayscale(self) -> Self:
        if self._color != cv.IMREAD_GRAYSCALE:
            self._data = cv.cvtColor(self._data, cv.COLOR_BGR2GRAY)
            self._color = cv.IMREAD_GRAYSCALE

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

 
    # Morphological operators

    
    @apply_on_copy
    def Erode(self, kernel_shape:str="rectangular", kernel_size:int=3, iterations:int=1) -> Self:
        """
        Applies erosion to the image.
        `kernel_shape`: Shape of the kernel. Can be: `rectangular` (default), `eliptical`, or `cross`.
        `kernel_size`: Size of the erosion kernel.
        `iterations`: Number of times the erosion is applied.
        """
        kernel = get_kernel(kernel_shape, kernel_size, self.GetBitDepth())
        self._data = cv.erode(self._data, kernel, iterations=iterations)
    

    @apply_on_copy
    def Dilate(self, kernel_shape:str="rectangular", kernel_size:int=3, iterations:int=1) -> Self:
        """
        Applies dilation to the image.
        `kernel_size`: Size of the dilation kernel.
        `iterations`: Number of times the dilation is applied.
        """
        kernel = get_kernel(kernel_shape, kernel_size, self.GetBitDepth())
        self._data = cv.dilate(self._data, kernel, iterations=iterations)

    
    @apply_on_copy
    def Opening(self, kernel_shape:str="rectangular", kernel_size:int=3, iterations:int=1) -> Self:
        """
        Applies opening (erosion followed by dilation) to the image.
        `kernel_size`: Size of the kernel used for the operations.
        """
        kernel = get_kernel(kernel_shape, kernel_size, self.GetBitDepth())
        self._data = cv.morphologyEx(self._data, cv.MORPH_OPEN, kernel, iterations=iterations)


    @apply_on_copy
    def Closing(self, kernel_shape:str="rectangular", kernel_size:int=3, iterations:int=1) -> Self:
        """
        Applies closing (dilation followed by erosion) to the image.
        `kernel_size`: Size of the kernel used for the operations.
        """
        kernel = get_kernel(kernel_shape, kernel_size, self.GetBitDepth())
        self._data = cv.morphologyEx(self._data, cv.MORPH_CLOSE, kernel, iterations=iterations)


    # Filters

    @apply_on_copy
    def GaussianBlur(self, kernel_size:tuple[int]=(5,5), sigmaX:int=0) -> Self:
        """
        Apply GaussianBlur on the image.

        `kernel_size`: size of Gausian blur kernel (matrix size)
        `sigmaX`: amount of used filter
        
        Return new image with applied filter.
        """
        self._data = cv.GaussianBlur(self._data, kernel_size, sigmaX)


    @apply_on_copy
    def AverageFilter(self, kernel_size:int=5) -> Self:
        """
        Applies an average filter to the image.
        `kernel_size`: Size of the averaging kernel.
        """
        self._data = cv.blur(self._data, (kernel_size, kernel_size))


    @apply_on_copy
    def MedianFilter(self, kernel_size:int=5) -> Self:
        """
        Applies a median filter to the image.
        `kernel_size`: Size of the kernel (must be odd).
        """
        self._data = cv.medianBlur(self._data, kernel_size)


    @apply_on_copy
    def BilateralFilter(self, diameter:int=9, sigma_color:int=75, sigma_space:int=75) -> Self:
        """
        Applies a bilateral filter to the image.
        `diameter`: Diameter of each pixel neighborhood.
        `sigma_color`: Filter sigma in color space.
        `sigma_space`: Filter sigma in coordinate space.
        """
        self._data = cv.bilateralFilter(self._data, diameter, sigma_color, sigma_space)
    

    # Gabor filter

    @apply_on_copy
    def GaborFilterSkimage(self, frequencies:list[float], thetas:list[float], sigma:float) -> Self:
        """
        Applies a Gabor filter using Scikit-Image's gabor_kernel function.

        `frequencies`: List of frequencies to control the number of wave cycles within the kernel.
                                    Higher frequencies detect finer patterns, while lower ones capture broader structures.
        `thetas`: List of orientations (in radians) to specify the direction of edge or pattern detection.
                                Different theta values detect edges or textures in various orientations.
        `sigma`: Controls the width of the Gaussian envelope in both x and y directions.
                        A larger sigma results in a broader filter that captures more context but loses fine detail.

        The function computes both real and imaginary parts of the Gabor response and combines them using magnitude.
        The result is a sum of the filtered responses across all frequencies and orientations, normalized to [0, 1].
        """
        filtered_images = []
        for theta in thetas:
            for frequency in frequencies:
                kernel = gabor_kernel(frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                filtered_real = convolve(self._data, np.real(kernel), mode='reflect')
                filtered_imag = convolve(self._data, np.imag(kernel), mode='reflect')
                filtered_magnitude = np.hypot(filtered_real, filtered_imag)
                filtered_images.append(filtered_magnitude)
        combined_image = np.sum([img for img in filtered_images], axis=0)
        self._data = combined_image / combined_image.max()


    @apply_on_copy
    def GaborFilterCV(self, kernel_size:int, lambd_values:list[float], theta_values:list[float], sigma:float, gamma:float=1, psi:float=0) -> Self:
        """
        Applies a Gabor filter using OpenCV's getGaborKernel function.
        
        `kernel_size`: Size of the Gabor kernel (width and height). Larger kernels capture broader features but miss fine details.
        `lambd_values`: List of wavelengths (spatial periods of the sinusoidal wave).
                                    Shorter wavelengths detect fine textures, while longer wavelengths capture larger features.
        `theta_values`: List of orientations (in radians) for detecting edges or patterns in specific directions.
                                    Different theta values detect features at various angles.
        `sigma`: Standard deviation of the Gaussian envelope. Controls how localized or broad the filter is.
                        Larger sigma values make the filter more sensitive to larger-scale features.
        `gamma`: Spatial aspect ratio, controlling the elongation of the kernel.
                                    A value less than 1 makes the kernel elongated in one direction, favoring directional detection.
        `psi`: Phase offset of the sinusoidal wave. A value of 0 creates a symmetric kernel, while non-zero values shift the wave phase.

        The function applies Gabor filtering using OpenCV's 2D convolution (filter2D), summing up the responses across all
        wavelengths and orientations, and normalizing the result to [0, 1].
        """
        filtered_images = []
        for lambd in lambd_values:
            for theta in theta_values:
                kernel = cv.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv.CV_32F)
                filtered_img = cv.filter2D(self._data, -1, kernel)
                filtered_images.append(filtered_img)
        combined_image = np.sum([img for img in filtered_images], axis=0)
        self._data = combined_image / combined_image.max()


    # Wavelet Transform

    def WaveletTransform(self, mode:str='haar') -> list[Self]:
        """
        Applies a 2D discrete wavelet transform (DWT) to the image data, decomposing it into
        four components: LL (approximation), LH (horizontal detail), HL (vertical detail), and HH (diagonal detail).
        The LL component is normalized for better visualization due to high intensity in 16-bit images.

        mode (str): The type of wavelet to use. Common options include:
            - 'haar': Simple, non-smooth wavelet (default).
            - 'dbN': Daubechies wavelets (e.g., 'db1' is equivalent to 'haar', 'db2', 'db3', etc.).
            - 'symN': Symlets, symmetric versions of Daubechies (e.g., 'sym2', 'sym3', etc.).
            - 'coifN': Coiflets, more symmetric with more vanishing moments (e.g., 'coif1', 'coif2', etc.).
            - 'biorN.N': Biorthogonal wavelets for symmetric, reversible transforms (e.g., 'bior1.3', 'bior2.2').
            - 'dmey': Discrete Meyer wavelet with good frequency localization.
        """
        imgs = ["LL", "LH", "HL", "HH"]
        LL, (LH, HL, HH) = pywt.dwt2(self._data, mode)

        # Normalize LL to 8-bit range (0-255)
        if self.GetBitDepth() == np.uint16:
            LL_normalized = 255 * (LL - np.min(LL)) / (np.max(LL) - np.min(LL))

        wavelet_result = []
        for nm, coe in zip(imgs, [LL_normalized, LH, HL, HH]):
            wavelet_result.append(Image(coe, f"{self.name}_Wavelet_{nm}"))
        return wavelet_result
    

    # Histogram Equalization

    @apply_on_copy
    def Normalize(self) -> Self:
        self._data = cv.normalize(self._data, None, 0, 65535, norm_type=cv.NORM_MINMAX)


    @apply_on_copy
    def HistogramEqualization(self) -> Self:
        """
        Applies histogram equalization to enhance the contrast of the image. Only on grayscale images!
        """
        if self._color != cv.IMREAD_GRAYSCALE:
            raise ValueError("Image.HistogramEqualization() the given image is not grayscale!")
        
        if self.GetBitDepth() == np.uint16:
            self._data = histogram_equalization_16_bit(self._data)
        else:
            self._data = cv.equalizeHist(self._data)


    @apply_on_copy
    def CLAHE_HistogramEqualization(self, max_clip:float=80.0, min_clip:float=2.0, tileSize:tuple=(8,8)) -> Self:
        """
        Contrast Limited Adaptive Histogram Equalization with adaptive clip limit.

        `max_clip`: Maximal threshold for contrast limiting.
        `min_clip`: Minimal threshold for contrast limiting.
        `tileSize`: Size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column.
        """
        brightness = np.mean(self._data) / 65535  # Scale brightness to range [0, 1]
        clip_limit = max(min_clip, min(max_clip, max_clip * (1 - brightness)))
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tileSize)
        self._data = clahe.apply(self._data)

    
    @apply_on_copy
    def WaveletDenoise(self, channel_axis:None|int=None) -> Self:
        """
        Apply wavelet denoise. Required normalization to [0, 1] and conversion back to 16-bit! 

        `multichannel`: If None, the image is assumed to be grayscale (single-channel). Otherwise, this parameter indicates which axis of the array corresponds to channels.
        """
        # Normalize the image to [0, 1] range for denoising
        image_float = self._data / 65535.0

        # The method='BayesShrink' parameter automatically adapts the amount of denoising based on the noise level, preserving details.
        # mode='soft' applies soft thresholding, which generally works well for smooth but noisy images.
        denoised_float = denoise_wavelet(image_float, method='BayesShrink', mode='hard', channel_axis=channel_axis, rescale_sigma=True)

        self._data = (denoised_float * 65535).astype(np.uint16)



    # Pyramids


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
