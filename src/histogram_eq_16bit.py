"""
This code was sourced from https://stackoverflow.com/questions/70442513/why-does-histogram-equalization-on-a-16-bit-image-show-a-strange-result
It is implementation of histogram equalization for 16-bit images inspired by OpenCV documentation: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
"""

import numpy as np

def histogram_equalization_16_bit(image:np.ndarray) -> np.ndarray:
    if image.dtype != np.uint16:
        raise ValueError(f"histogram_equalization_16_bit(): given image has not 16-bit depth, dtype is: {image.dtype}")

    hist, _ = np.histogram(image.flatten(), 65536, [0, 65536])  # Collect 16 bits histogram (65536 = 2^16).
    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)  # Find the minimum histogram value (excluding 0)
    cdf_m = (cdf_m - cdf_m.min())*65535/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint16')

    # Now we have the look-up table...
    return cdf[image]