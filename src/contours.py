import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

def find_contours(img):
    """
    This method finds the contours of the rod in the given image.

    Parameters
    ----------
    img : numpy.ndarray
        Image to process.

    Returns
    -------
    numpy.ndarray
        Contours of the rod.
    numpy.ndarray
        Hierarchy of the contours.
    numpy.ndarray
        External contours of the rod.
    numpy.ndarray
        Holes contours of the rod.
    """

    working_img = img.copy()

    # Get the contours of the rod, RETR_CCOMP retrieves all of the contours and organizes
    # them into a two-level hierarchy
    _, contours, hierarchy = cv2.findContours(
        working_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    # Get the external contours, those are the contours of the rod
    # hierarchy = [next, prev, child, parent]
    ext_contours = []
    for j in range(len(hierarchy[0])):
        if hierarchy[0][j][3] == -1:
            ext_contours.append(contours[j])

    # Holes contours
    holes_contours = []
    for j in range(len(hierarchy[0])):
        if hierarchy[0][j][3] != -1:
            holes_contours.append(contours[j])

    # Filter small contours by area
    # ext_contours = [contour for contour in ext_contours if cv2.contourArea(contour) > 1000]
    holes_contours = [contour for contour in holes_contours if cv2.contourArea(contour) > 100]


    return contours, hierarchy, ext_contours, holes_contours