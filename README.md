# BIO-FPVE

For more information see [docs](doc/docs.pdf). Dependencies are listed in [requirements.txt](requirements.txt).


This project aims to enhance blood vessel structures in image sets using advanced fusion and image processing techniques. It integrates preprocessing, fusion, and postprocessing to improve vessel visibility.

## Project Structure
The main components of the project are located in the `src` folder:

- `fusion.py`: Implements image fusion methods.
- `histogram_eq_16bit.py`: Performs histogram equalization on 16-bit images.
- `image.py`: Contains a class for individual image manipulation.
- `image_set.py`: Contains a class for processing image sets.
- `improved_hdr.py`: Methods for enhanced HDR image processing.
- `main.py`: Main script to execute the project.
- `pyramids.py`: Algorithms for Laplacian and Gaussian pyramids.
- `utils.py`: Auxiliary functions.

## Methods Used
The project employs three key phases:

### 1. Preprocessing
- **Histogram Equalization** (`histogram_eq_16bit.py`): Balances histograms to enhance contrast.
- **Gaussian Blur**: Reduces noise and smoothens images.
- **Morphological Operations**: Emphasizes structures like blood vessels.

### 2. Image Fusion
- **Wavelet Fusion**: Uses wavelet transform for coefficient combination.
- **Laplacian Pyramid Fusion**: Combines Laplacian pyramids of images.
- **PCA Fusion**: Employs Principal Component Analysis.
- **Exposure Fusion**: Combines images using contrast, saturation, and exposure.
- **Pixel Averaging**: Averages pixel values across images.
- **Improved HDR (IHDR)**: Extracts details through Gaussian and Laplacian pyramid fusion.

### 3. Postprocessing (for IHDR only)
- **Normalization**: Restores intensity range.
- **CLAHE Histogram Equalization**: Enhances local contrast.
- **Morphological Operations**: Reduces noise and highlights structures.

## Examples of Usage
Run the main script with various parameters for different functionalities:

1. **Default processing:**
   ```bash
   python main.py "data/input_images" all --save "data/output_results"
    ```

2. **Specific fusion methods:**
    ```bash
    python main.py "data/input_images" laplacian pca --save "data/output_results"
    ```

3. **With mask processing:**
    ```bash
    python main.py "data/input_images" exposure --mask "data/mask.png" --save "data/output_results"
    ```

4. **Custom steps (fusion only):**
    ```bash
    python main.py "data/input_images" wavelet --steps f --save "data/output_results"
    ```

5. **Preprocessing only:**
    ```bash
    python main.py "data/input_images" wavelet --steps p --save "data/preprocessed_images"
    ```

6. **Best results setup:**
    ```bash
    python main.py "data/input_images" ihdr --steps pfp --save "data/preprocessed_images"
    ```