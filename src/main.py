"""
@file: image.py
@brief: FPVE main
@author: Michal Ľaš (xlasmi00), Tomáš Bártů (xbartu11)
@date: 10.11.2024
"""

from utils import load_image_set, load_image
from image import Image
from image_set import ImageSet
from fusion import exposure_fusion, average_fusion, laplacian_pyramid_fusion, pca_fusion
from cv2 import CV_8U




def fusion(image_set: ImageSet, method: str = 'wavelet', **kwargs) -> Image:
    if method == 'wavelet':
        fused_image = image_set.WaveletFusion(
            wavelet=kwargs.get('wavelet', 'sym5'),
            level=kwargs.get('level', 5),
            combine_method=kwargs.get('combine_method', 'mean')
        )
    elif method == 'exposure':
        fused_image = exposure_fusion(image_set)
    elif method == 'average':
        fused_image = average_fusion(image_set)
    elif method == 'laplacian':
        fused_image = laplacian_pyramid_fusion(image_set, max_level=9)
    elif method == 'pca':
        fused_image = pca_fusion(image_set)
    else:
        raise ValueError("Unsupported fusion method. Use 'wavelet', 'exposure', 'average', or 'laplacian'.")
    
    return fused_image


def preprocessing(image_set:ImageSet, mask:None|Image=None) -> ImageSet:
    # Preprocess
    for img in image_set:
        img.ScaleBack()
        img.CLAHE_HistogramEqualization(80, 2, mask, False, tileSize=(20,20))
        img.GaussianBlur((5,5))
        img.Dilate(kernel_size=3, kernel_shape='eliptical', iterations=3)
        img.Opening(kernel_size=3, kernel_shape='eliptical', iterations=3)
        img.CLAHE_HistogramEqualization(80, 40, mask, tileSize=(20,20))
        img.ApplyMask(mask)
        
    
    # Filter too bright or too dark images
    mask_mean = img.GetData()[mask.GetData() == 1].mean()
    filtered_set = [img for img in image_set if (img.GetData().max() * 0.2) < img.GetData()[mask.GetData() == 1].mean() < (img.GetData().max() * 0.8)]
    
    return image_set



def process_mask(raw_mask:Image) -> Image:
    raw_mask.ScaleBack()
    raw_mask.CLAHE_HistogramEqualization(80, 2, tileSize=(20,20))
    raw_mask.Opening('eliptical', 8, iterations=5)
    raw_mask.GaussianBlur()
    raw_mask.ScaleBack(255, CV_8U)
    raw_mask.ThresholdBinary(160, 255)
    raw_mask.Erode('eliptical', 6)
    raw_mask.ScaleBack(1, CV_8U)
    return raw_mask



def main():

    # Load mask (and process)
    mask_image_path = '../samples/nir01_5.0_0.png'
    mask_image = load_image(mask_image_path)
    mask:Image = process_mask(mask_image)

    # Load images
    folder_path = '../samples/for-testing/002-r-1'
    image_set = load_image_set(folder_path)

    # Preprocessing
    image_set = preprocessing(image_set, mask)

    # Fusion
    results = ImageSet([], "Final results")
    fusion_types = ['wavelet', 'exposure', 'average', 'laplacian', 'pca']

    for fusion_type in fusion_types:
        fused_image = fusion(image_set, method=fusion_type)
        fused_image.name = f'{fusion_type}'
        results.Append(fused_image)
        fused_image.Save(f'../img/{fused_image.name}.png')


    results.SlideShow()


if __name__=="__main__":
    main()