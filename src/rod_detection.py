import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from src.preprocess import apply_filter, binarize


def get_connected_component(rod_info, i):
    """
    Returns the connected component with the given index.

    Parameters
    ----------
    rod_info : dict
        Dictionary containing information about the rods.
    i : int
        Index of the connected component to return.

    Returns
    -------
    numpy.ndarray
        Connected component with the given index.
    """
    mask = np.zeros_like(rod_info["labels"], dtype=np.uint8)
    mask[rod_info["labels"] == i] = 255
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


def get_ellipse_axis(contours, barycenter):
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
            mu[j]["m10"] / (mu[j]["m00"] + 1e-5),
            mu[j]["m01"] / (mu[j]["m00"] + 1e-5),
        )
    return mc


def findMER(contours):
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


def detect_rods_blob(image, visualize=True):
    """
    This method detects the rods objects in the given image using blob detection and
    counts the number of holes in each rod.

    Parameters
    ----------
    img : numpy.ndarray
        Image to process.

    Returns
    -------
    dict
        Dictionary containing information about the rods.
    """
    img = image.copy()
    rod_info = {
        "num_labels": None,
        "centroids": None,
        "labels": None,
        "number_of_rods": None,
        "contours": [],
        "hierarchy": [],
        "number_of_holes": [],
        "rod_type": [],
        "coord": [],
        "dim": [],
        "area": [],
        "angle": [],
        "barycenter": [],
        "length": [],
        "width": [],
    }

    img = apply_filter(img, sigma=1.0)
    binary_img = binarize(img)

    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gae57b028a2b2ca327227c2399a9d53241
    # Find the connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=4
    )

    rod_info["num_labels"] = num_labels
    rod_info["centroids"] = centroids
    rod_info["labels"] = labels
    rod_info["number_of_rods"] = num_labels - 1  # Remove background
    print("Number of rods found (CC): {}".format(num_labels - 1))

    print("Processing connected components individually...\n")
    for i in range(1, num_labels):
        # Loop over the connected components, 0 is the background
        print("Processing rod {}...".format(i))
        # Get the masked image, now the image will contain only one rod
        comp = get_connected_component(rod_info, i)

        # Those are the statistics about the connected component
        (cX, cY) = rod_info["centroids"][i]
        x, y, w, h, area = stats[i]
        rod_info["coord"].append((x, y))
        rod_info["dim"].append((w, h))
        rod_info["area"].append(area)
        print("Rod {}: area (CC): {}".format(i, area))

        # Get the contours of the rod, RETR_CCOMP retrieves all of the contours and organizes
        # them into a two-level hierarchy
        _, contours, hierarchy = cv2.findContours(
            comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        rod_info["contours"].append(contours)
        rod_info["hierarchy"].append(hierarchy)

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

        # A hole has a parent contour but no child contour
        n_holes = sum(
            1
            for j in range(hierarchy[0].shape[0])
            if hierarchy[0][j][2] == -1 and hierarchy[0][j][3] > -1
        )

        print("Rod {}: number of holes: {}".format(i, n_holes))
        rod_info["number_of_holes"].append(n_holes)

        # Rod type detection
        if rod_info["number_of_holes"][i - 1] == 1:
            rod_type = "A"
        elif rod_info["number_of_holes"][i - 1] == 2:
            rod_type = "B"
        else:
            rod_type = "Unknown"

        rod_info["rod_type"].append(rod_type)
        print("Rod {}: type: {}".format(i, rod_type))

        # Get the minimum enclosing rectangle of the rod
        box, ((rX, rY), (r_width, r_height), angle) = findMER(ext_contours[0])

        # Get the length of the rod
        length = max(r_width, r_height)
        rod_info["length"].append(length)
        print("Rod {}: length (of the Major Axis): {}".format(i, length))

        # Get the width of the rod
        width = min(r_width, r_height)
        rod_info["width"].append(width)
        print("Rod {}: width (of the Minor Axis): {}".format(i, width))

        # The angle of the rod is the angle of the minimum enclosing rectangle
        # The angle is modulo pi
        angle = angle % 180
        rod_info["angle"].append(angle)
        print("Rod {}: angle: {}".format(i, angle))

        # Get the central moments of the rod, those are invariant to translation and scaling
        # To get invariance to scaling nu20, nu11, nu02, nu30, nu21, nu12, nu03 are used
        mu = get_moments(ext_contours)
        print("Rod {}: moments: {}".format(i, mu))

        # Get the mass centers of the rod
        mc = get_mass_center(ext_contours, mu)
        rod_info["barycenter"].append(mc)
        print("Rod {}: mass center(s): {}".format(i, mc))

        if visualize:
            # Plot the connected component
            output = binary_img.copy()
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
            plt.figure(figsize=(5, 5))

            # Draw the center mass
            for m in mc:
                cv2.circle(output, (int(m[0]), int(m[1])), 4, (255, 0, 255), -1)
                cv2.putText(
                    output,
                    str(i),
                    (int(cX) - 12, int(cY) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            # Draw the MER
            cv2.drawContours(output, [box], -1, (0, 255, 0), 1)

            # Draw the contours
            cv2.drawContours(output, ext_contours, -1, (0, 0, 255), 1)
            cv2.drawContours(output, holes_contours, -1, (255, 0, 0), 1)

            plt.imshow(output, cmap="gray")
            plt.title("Rod {}".format(i))
            plt.show()

        print("-" * 50)
    return rod_info


def detect_rods(images: list, names: List[str], visualize=True):
    results = {}
    for image, name in zip(images, names):
        print("Processing image: {}".format(name))
        rod_info = detect_rods_blob(image, visualize=visualize)
        results[name] = rod_info

    return results
