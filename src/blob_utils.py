################################################################################
# BLOB UTILS
################################################################################

import numpy as np
import cv2

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

def find_ellipsis(contours, barycenter: tuple):
    """
    Returns the major and minor axis of the given contours. The axis both
    pass through the barycenter of the contours.

    Parameters
    ----------
    contours : list
        List of contours to process.
    barycenter : tuple
        Barycenter of the contours.

    Returns
    -------
    tuple
        the fitted ellipse, the major axis and the minor axis."""
    contours = np.array(contours[0])
    # Fit an ellipse around the contour
    ellipse = cv2.fitEllipse(contours)
    _, (MA, ma), angle = ellipse

    # Convert the angle to the range [0, 180]
    angle = angle % 180

    bX, bY = barycenter

    print(MA, ma)

    # Extrema of the major axis
    MA_x1 = int(bX + (MA / 2) * np.cos(np.deg2rad(angle)))
    MA_y1 = int(bY + (MA / 2) * np.sin(np.deg2rad(angle)))
    MA_x2 = int(bX - (MA / 2) * np.cos(np.deg2rad(angle)))
    MA_y2 = int(bY - (MA / 2) * np.sin(np.deg2rad(angle)))

    # Extrema of the minor axis
    ma_x1 = int(bX + (ma / 2) * np.sin(np.deg2rad(angle)))
    ma_y1 = int(bY - (ma / 2) * np.cos(np.deg2rad(angle)))
    ma_x2 = int(bX - (ma / 2) * np.sin(np.deg2rad(angle)))
    ma_y2 = int(bY + (ma / 2) * np.cos(np.deg2rad(angle)))

    return (
        ellipse,
        ((MA_x1, MA_y1), (MA_x2, MA_y2)),
        ((ma_x1, ma_y1), (ma_x2, ma_y2)),
    )

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
            int(mu[j]["m10"] / (mu[j]["m00"] + 1e-5)),
            int(mu[j]["m01"] / (mu[j]["m00"] + 1e-5)),
        )
    return mc

def find_MER(contours):
    """
    Returns the minimum enclosing rectangle of the given contour.

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

    return box, ((rX, rY), (width, height), angle)

def get_major_axis(theta, mc):
    """
    Returns the major axis of the given object.

    Parameters
    ----------
    theta : float
        Angle of the major axis wrt the horizontal axis.
    mc : tuple
        Mass center of the object.

    Returns
    -------
    tuple
        Coefficients of the major axis equation.
    """
    # Equation of the major axis
    a = -np.sin(theta)
    b = -np.cos(theta)
    c = np.cos(theta) * mc[1] + np.sin(theta) * mc[0]
    return (a, b, c)