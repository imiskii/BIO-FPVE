"""
@file: image.py
@brief: FPVE main
@author: Michal Ľaš (xlasmi00), Tomáš Bártů (xbartu11)
@date: 10.11.2024
"""

import argparse
from utils import load_image_set, load_image
from image import Image
from image_set import ImageSet
from fusion import exposure_fusion, average_fusion, laplacian_pyramid_fusion, pca_fusion
from cv2 import CV_8U


FUSION_METHODS:list[str] = ['wavelet', 'exposure', 'average', 'laplacian', 'pca']


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finger Vein Enhancer")

    parser.add_argument("ifolder", nargs=1, type=str, help="Path to the folder with images.")
    parser.add_argument("methods", nargs='+', type=str, choices=FUSION_METHODS + ['all'], help="Select fusion method.") # TODO: write possible methods (include 'all')
    parser.add_argument("--save", required=False, nargs=1, type=str, default=None, help="Path to the folder where results will be saved.")
    parser.add_argument("--mask", required=False, nargs=1, type=str, default=None, help="Path to the mask image.")
    parser.add_argument("--proc_mask", required=False, action="store_true", default=False, help="Bool flag to preprocess mask before using it.")

    return parser


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
        img.CLAHE_HistogramEqualization(80, 40, mask, True, tileSize=(20,20))
        if mask is not None:
            img.ApplyMask(mask)
        
    
    # Filter too bright or too dark images
    filtered_set = [img for img in image_set if (img.GetData().max() * 0.2) < img.GetData()[mask.GetData() == 1].mean() < (img.GetData().max() * 0.8)]
    
    return ImageSet(filtered_set, "Input_images_preprocessed")



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

    # Parse arguments
    parser = argument_parser()
    args = parser.parse_args()

    folder_path = args.ifolder[0]
    mask_image_path = args.mask[0]
    selected_fusion_methods = args.methods if args.methods[0] != "all" else FUSION_METHODS
    proc_mask = args.proc_mask
    save_folder = args.save

    print(f"""Start processing...
Folder path: {folder_path}
Selected methods: {selected_fusion_methods}
Mask path: {mask_image_path}
Mask processing: {proc_mask}
Save folder: {save_folder}\n""")

    if save_folder is not None:
        save_folder = save_folder[0]
        if save_folder[-1] != "/":
            save_folder += "/"


    # Load mask (and process)
    mask_image = None
    if mask_image_path is not None:
        mask_image = load_image(mask_image_path)
        mask_image.name = f"mask"
        if proc_mask:
            mask:Image = process_mask(mask_image)

    # Load images
    image_set:ImageSet = load_image_set(folder_path)

    # Preprocessing
    image_set:ImageSet = preprocessing(image_set, mask)

    # Fusion
    results = ImageSet([], "Final results")

    for fusion_type in selected_fusion_methods:
        fused_image = fusion(image_set, method=fusion_type)
        fused_image.name = f'{fusion_type}'
        results.Append(fused_image)
        if save_folder is not None:
            fused_image.Save(f'{save_folder}{fused_image.name}.png')


    results.SlideShow()


if __name__=="__main__":
    main()