import numpy as np
import cv2
import math
from typing import Tuple

def get_connected_component(labels, i):
    """
    Returns the connected component with the given index.

    Parameters
    ----------
    labels : numpy.ndarray
        Array of labels of the connected components.
    i : int
        Index of the connected component to return.

    Returns
    -------
    numpy.ndarray
        Connected component with the given index.
    """
    mask = np.zeros_like(labels, dtype=np.uint8)
    mask[labels == i] = 255
    return mask

def get_moments(contours):
    """
    Returns the moments of the given contours.

    Central moments: mu20, mu11, mu02, mu30, mu21, mu12, mu03
    https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html

    Parameters
    ----------
    contours : list
        List of contours to process.

    Returns
    -------
    list
        List of moments up to the third order."""
    mu = [None] * len(contours)
    for j in range(len(contours)):
        mu[j] = cv2.moments(contours[j])
    return mu

def get_mass_center(contours, mu):
    """
    Returns the mass center (or barycenter in this case) of the
    given contours representing an object.

    C_x = M10 / M00
    C_y = M01 / M00

    Parameters
    ----------
    contours : list
        List of contours to process.

    Returns
    -------
    mu : list
        List of moments of the contours."""
    mc = [None] * len(contours)
    for j in range(len(contours)):
        mc[j] = (
            # add 1e-5 to avoid division by zero
            int(mu[j]["m10"] / (mu[j]["m00"])),
            int(mu[j]["m01"] / (mu[j]["m00"])),
        )
    return mc

def find_MER(contours):
    """
    Returns the minimum enclosing rectangle of the given contour. It corrects the angle
    to be in the range of (0, 180] and the width and height to be the shortest and longest
    side of the rectangle respectively.

    Parameters
    ----------
    contours : numpy.ndarray
        Contour to process.

    Returns
    -------
    numpy.ndarray
        Array of points representing the minimum enclosing rectangle.

    tuple
        Tuple containing the center, width, height and angle of the rectangle.
    """
    contour_points = np.array(contours)

    # Compute the minimum enclosing rectangle
    rectangle = cv2.minAreaRect(contour_points)
    ((rX, rY), (width, height), angle) = rectangle

    # Get the box points of the rectangle
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)

    corr_height = height
    corr_width = width

    # Angle and dim correction: angle returned by the method is [90, 0)
    if width < height:
        corr_angle = 90 - angle
    else:
        corr_height = width
        corr_width = height
        corr_angle = -angle

    return box, ((rX, rY), (corr_width, corr_height), corr_angle)

def get_axis_from_fitted_obj(angle: float, center: Tuple[float, float], diam: Tuple[float, float]):
    """
    Returns the major and minor axis of the given fitted object.

    Parameters
    ----------
    angle : float
        Angle of the fitted object.
    center : tuple
        Center of the fitted object.
    diam : tuple
        Diameters of the fitted object. In case of a rectangle, the diameters are simply the width and height.

    Returns
    -------
    tuple
        Major axis of the fitted object.
    tuple
        Minor axis of the fitted object.
    """
    # Major axis
    rmajor = max(diam[0], diam[1]) / 2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90

    x1 = center[0] + math.cos(math.radians(angle)) * rmajor
    y1 = center[1] + math.sin(math.radians(angle)) * rmajor
    x2 = center[0] + math.cos(math.radians(angle+180)) * rmajor
    y2 = center[1] + math.sin(math.radians(angle+180)) * rmajor
    MA = tuple(np.int0([x1, y1, x2, y2]))

    # Minor Axis
    rminor = min(diam[0], diam[1]) / 2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90

    x1 = center[0] + math.cos(math.radians(angle)) * rminor
    y1 = center[1] + math.sin(math.radians(angle)) * rminor
    x2 = center[0] + math.cos(math.radians(angle+180)) * rminor
    y2 = center[1] + math.sin(math.radians(angle+180)) * rminor
    ma = tuple(np.int0([x1, y1, x2, y2]))

    return MA, ma

def get_axes_by_ellipse(contours: np.ndarray):
    """
    Returns the major and minor axis by fitting an ellipse to the given contour.

    Parameters
    ----------
    contours : numpy.ndarray
        Contour to process.
    
    Returns
    -------
    tuple
        Major axis of the fitted object.
    tuple
        Minor axis of the fitted object.
    float
        Angle of the major axis.
    """
    # fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree 
    ellipse = cv2.fitEllipse(contours)
    (xc, yc), (d1, d2), angle = ellipse

    MA, ma = get_axis_from_fitted_obj(angle, (xc, yc), (d1, d2))
    # (MA_x1, MA_y1, MA_x2, MA_y2) = MA
    # (ma_x1, ma_y1, ma_x2, ma_y2) = ma

    return MA, ma

def compute_major_axis(mu, mc):
    """
    Returns the major axis by computing the covariance matrix on the second
    order central moments.

    Parameters
    ----------
    mu : dict
        Dictionary of moments of the contour.
    mc : tuple
        Tuple containing the center of the contour.

    Returns
    -------
    numpy.ndarray
        Array of coefficients of the major axis.
    
    float
        Angle of the major axis wrt the horizontal axis.
    """

    # Calculate the covariance matrix
    u20 = mu['mu20'] / mu['m00']
    u11 = mu['mu11'] / mu['m00']
    u02 = mu['mu02'] / mu['m00']

    cov_matrix = np.array([[u20, u11],
                            [u11, u02]])

    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Determine the major axis coefficients
    major_axis_coeffs = sorted_eigenvectors[:, 0]

    # Calculate the angle of the major axis
    angle = np.arctan(major_axis_coeffs[0] / major_axis_coeffs[1]) + np.pi / 2

    # Optionally normalize the coefficients
    major_axis_coeffs /= np.linalg.norm(major_axis_coeffs)

    return major_axis_coeffs, angle

def get_holes_diameter(int_contour):
    """
    Returns the diameter of the holes in the given contour.

    Parameters
    ----------
    int_contour : numpy.ndarray
        Contour to process.
    
    Returns
    -------
    list
        List of diameters of the holes in the contour.
    """
    diameter = []
    for hole in int_contour:
        _, radius = cv2.minEnclosingCircle(hole)
        diameter.append(2 * radius)
    return diameter