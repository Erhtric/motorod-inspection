import cv2
import numpy as np


def apply_gaussian_filter(img, sigma):
    """
    This method applies a Gaussian filter to the given image.

    Parameters
    ----------
    img : numpy.ndarray
        Image to process.
    sigma : float
        Standard deviation for Gaussian kernel.
    kernel_size : int
        Use rule of thumb to compute the size.

    Returns
    -------
    numpy.ndarray
        Filtered image.
    """
    kernel_size = int(np.ceil((3 * sigma)) * 2 + 1)

    print(
        "Gaussian filter parameters --- sigma: {}, kernel size: {}".format(
            sigma, kernel_size
        )
    )
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    return img


def binarize(img, method=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU):
    """
    This method binarizes the given image.

    Parameters
    ----------
    img : numpy.ndarray
        Image to process.
    method : int
        Thresholding method. Default is cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU.

    Returns
    -------
    numpy.ndarray
        Binarized image.
    """
    # To create a binary image, we need to threshold the image
    _, binary_img = cv2.threshold(img, 0, 255, method)
    return binary_img
