import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from src.preprocess import apply_filter, binarize
from src.blob_utils import get_connected_component, get_moments, get_mass_center, find_MER, get_major_axis


def get_width_at_mass_center(img, ext_contours, theta, mc, visualize=True):
    """Compute the width of the object at the barycenter
    
    Parameters
    ----------
    img : numpy.ndarray
        Image to process.
    ext_contours : list
        List of contours to process.
    theta : float
        Angle of the major axis of the object.
    mc : tuple
        Tuple containing the mass center of the object.
        
    Returns
    -------
    width_barycenter : float
        Width of the object at the barycenter.
    """
    def signed_distance(a, b, c, i, j):
        """Compute the signed distance of a point (i, j) from the line ax + by + c = 0"""
        return (a*j + b*i + c) / np.sqrt(a*a + b*b)
    
    a, b, c = get_major_axis(theta, mc)

    # create a binary image representing the contour of the object
    contour_img = np.zeros_like(img)
    cv2.polylines(contour_img, ext_contours, True, 1, 1)

    # Iterate over the image to get the signed distances
    left_points = []
    right_points = []
    for i in range(contour_img.shape[0]):
        for j in range(contour_img.shape[1]):
            if contour_img[i, j] == 1:
                d = signed_distance(a, b, c, i, j)

                # Compute the euclidean distance with respect to the barycenter and assign the sign
                d = np.sqrt((j - mc[0]) ** 2 + (i - mc[1]) ** 2) * np.sign(d)

                # Create two lists of points, one for the points on the left (positive) and one for the points on the right (negative)
                if d > 0:
                    left_points.append((j, i, d))
                elif d < 0:
                    right_points.append((j, i, d))

    # Find the points with the minimum distance on the left and on the right
    left_points = np.array(left_points)
    right_points = np.array(right_points)

    # Compute the absolute minimum distance
    min_left_idx = np.argmin(np.abs(left_points[:, 2]), axis=0)
    p_left = np.int0(left_points[min_left_idx][:2])
    min_right_idx = np.argmin(np.abs(right_points[:, 2]), axis=0)
    p_right = np.int0(right_points[min_right_idx][:2])

    width_barycenter = np.sqrt((p_left[0] - p_right[0]) ** 2 + (p_left[1] - p_right[1]) ** 2)

    if visualize:
        # Draw the points
        output = img.copy()
        output = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2RGB)

        cv2.circle(output, tuple(p_left), 5, (255, 0, 0), -1)       # red
        cv2.circle(output, tuple(p_right), 5, (0, 0, 255), -1)      # blue

        # Given a,b,c draw the line ax + by + c = 0
        def draw_line(a, b, c, img, color=(255, 255, 255)):
            if b == 0:
                cv2.line(img, (0, int(-c / a)), (img.shape[1], int(-c / a)), color, 1)
            else:
                cv2.line(img, (0, int(-c / b)), (img.shape[1], int((-a * img.shape[1] - c) / b)), color, 1,)

        draw_line(a, b, c, output, (255, 0, 0))
        return width_barycenter, output

    return width_barycenter
    
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

    img = apply_filter(img, sigma=0.5)
    binary_img = binarize(img)
    output = None

    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gae57b028a2b2ca327227c2399a9d53241
    # Find the connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
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
        comp = get_connected_component(rod_info["labels"], i)

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
        box, ((rX, rY), (r_width, r_height), _ ) = find_MER(contours[0])

        # Get the length of the rod
        length = max(r_width, r_height)
        rod_info["length"].append(length)
        print("Rod {}: length (MER): {}".format(i, length))

        # Get the width of the rod
        width = min(r_width, r_height)
        rod_info["width"].append(width)
        print("Rod {}: width (MER): {}".format(i, width))

        # Get the central moments of the rod, those are invariant to translation and scaling
        # To get invariance to scaling nu20, nu11, nu02, nu30, nu21, nu12, nu03 are used
        mu = get_moments(ext_contours)

        # Get the mass centers of the rod
        mc = get_mass_center(ext_contours, mu)
        rod_info["barycenter"].extend(mc)
        print("Rod {}: mass center(s): {}".format(i, mc))

        # Get the orientation of the rod modulo pi, namely the angle between the major axis of the
        # horizontal axis of the image
        angle = 0.5 * np.arctan((2 * mu[0]['nu11']) / (mu[0]['nu02'] - mu[0]['nu20'])) + np.pi / 2
        print("Rod {}: angle (mod pi): {}".format(i, angle))
        rod_info["angle"].append(angle)

        # Compute the width of the rod at the mass center
        width_mc, output = get_width_at_mass_center(binary_img, ext_contours[0], angle, mc[0], visualize=visualize)
        print("Rod {}: width at mass center: {}".format(i, width_mc))

        if visualize:
            # Plot the connected component
            if output is None:
                output = binary_img.copy()
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

                      # Draw the center mass
            for m in mc:
                cv2.circle(output, (int(m[0]), int(m[1])), 4, (255, 0, 255), -1)
                cv2.putText(
                    output,
                    str(i),
                    (int(cX) - 20, int(cY) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            # Draw the MER
            cv2.drawContours(output, [box], -1, (0, 255, 0), 1)

            # Draw the major and minor axis
            # cv2.line(output, tuple(P1), tuple(P2), (255, 0, 0), 2)
            # cv2.line(output, tuple(P3), tuple(P4), (0, 255, 255), 2)

            # Draw the contours
            cv2.drawContours(output, ext_contours, -1, (255, 255, 255), 1)
            cv2.drawContours(output, holes_contours, -1, (255, 255, 255), 1)

            plt.figure(figsize=(7, 7))
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
