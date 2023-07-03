import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from src.preprocess import apply_gaussian_filter, binarize
from src.blob import get_axes_by_ellipse, compute_major_axis, get_connected_component, get_moments, get_mass_center, find_MER, get_holes_diameter
from src.contours import find_contours

def get_width_at_mass_center(img, ext_contours, mu, mc):
    """Compute the width of the object at the barycenter
    
    Parameters
    ----------
    img : numpy.ndarray
        Image to process.
    ext_contours : list
        List of contours to process.
    theta : float
        Angle of the major axis of the object.
    mu : list
        List of moments of the contours.
    mc : tuple
        Tuple containing the mass center of the object.
        
    Returns
    -------
    width_barycenter : float
        Width of the object at the barycenter.
    
    tuple
        Tuple containing the points on the left and on the right of the object.
    """
    # Compute the signed distance from the major axis  
    def signed_distance(MA_coeffs, mc, i, j):
        vect = np.array([j, i]) - mc
        return np.dot(MA_coeffs, vect)

    MA_coeffs, angle = compute_major_axis(mu[0], mc)

    # create a binary image representing the contour of the object
    contour_img = np.zeros_like(img)
    cv2.polylines(contour_img, ext_contours, True, 1, 1)

    # Iterate over the image to get the signed distances
    left_points = []
    right_points = []
    for i in range(contour_img.shape[0]):
        for j in range(contour_img.shape[1]):
            if contour_img[i, j] == 1:
                d = signed_distance(MA_coeffs, mc, i, j)

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

    return width_barycenter, (p_left, p_right)

def detect_contact_points(img):
    """
    This method detects the contact points between the rods.

    The code is based on the following answer:
        https://answers.opencv.org/question/87583/detach-blobs-with-a-contact-point/

    Parameters
    ----------
    img : numpy.ndarray
        Binary Image to process.

    Returns
    -------
    contact_points : list
        List of contact points.
    """
    _, contours, hierarchy = cv2.findContours(img, 2, cv2.CHAIN_APPROX_SIMPLE)

    contact_points = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 1500:
            # Approximate the contour with a polygon
            # contour = cv2.approxPolyDP(contour, 2, True)
            contoursHull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, contoursHull)

            for j in range(defects.shape[0]):
                defpoint = defects[j][0]
                pt = tuple(contour[defpoint[2]][0])  # Get defect point
                r3x3 = (pt[0] - 2, pt[1] - 2, 5, 5)  # Create 5x5 Rect from defect point

                # Make sure the rect is within the image bounds
                # r3x3 = (max(r3x3[0], 0), max(r3x3[1], 0), min(r3x3[2], img.shape[1]), min(r3x3[3], img.shape[0]))

                non_zero_pixels = np.count_nonzero(img[r3x3[1]:r3x3[1]+r3x3[3], r3x3[0]:r3x3[0]+r3x3[2]])
                if non_zero_pixels > 17:
                    contact_points.append(pt)

    return contact_points

def separate_rods(img):
    """
    This method separates the rods in the given image.

    Parameters
    ----------
    img : numpy.ndarray
        Binary Image to process.

    Returns
    -------
    img : numpy.ndarray
        Binary Image with the rods separated.
    """
    contact_points = detect_contact_points(img)

    # Draw a black line between nearby contact points
    for i in range(len(contact_points)):
        for j in range(i + 1, len(contact_points)):
            if np.linalg.norm(np.array(contact_points[i]) - np.array(contact_points[j])) < 20:
                cv2.line(img, contact_points[i], contact_points[j], 0, 2)

    return img
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")

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
    output = image.copy()
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)


    rod_info = {
        "num_labels": None,
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
        "width_at_barycenter": [],  
        "holes_barycenter": [],     
        "holes_diameter": [],           
    }

    # Apply a Gaussian filter to the image and binarize it using a threshold
    img = apply_gaussian_filter(img, sigma=1)
    binary_img = binarize(img)

    # binary_img = separate_rods(binary_img)

    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gae57b028a2b2ca327227c2399a9d53241
    # Find the connected components in the binary image
    num_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )

    rod_info["number_of_rods"] = num_labels - 1  # Remove background
    rod_info["num_labels"] = num_labels
    rod_info["labels"] = labels

    # # Filter out in the labels the small components
    # min_area = 500
    # rod_info["labels"] = np.zeros_like(labels)
    # count = 1
    # for i in range(1, num_labels):
    #     if stats[i][4] > min_area:
    #         rod_info["labels"][labels == i] = count
    #         count += 1

    # stats = stats[stats[:, 4] > minArea]
    # rod_info["num_labels"] = stats.shape[0]
    # rod_info["number_of_rods"] = rod_info["num_labels"] - 1  # Remove background

    print("Number of rods found (CC): {}".format(rod_info["number_of_rods"]))

    print("Processing connected components individually...\n")
    for i in range(1, rod_info["num_labels"]):
        # Loop over the connected components, 0 is the background
        print("Processing rod {}...".format(i))
        # Get the masked image, now the image will contain only one rod
        comp = get_connected_component(rod_info["labels"], i)

        _, _, _, _, area = stats[i]
        rod_info["area"].append(area)

        print("Centroid (CC): {}".format(tuple(np.int0(centroid[i]))))
        print("Area (CC): {}".format(area))

        contours, hierarchy, ext_contours, holes_contours = find_contours(comp)
        rod_info["contours"].append(contours)
        rod_info["hierarchy"].append(hierarchy)

        # A hole has a parent contour but no child contour
        n_holes = sum(
            1
            for j in range(hierarchy[0].shape[0])
            if hierarchy[0][j][2] == -1 and hierarchy[0][j][3] > -1
        )

        print("Number of holes: {}".format(n_holes))
        rod_info["number_of_holes"].append(n_holes)

        # Rod type detection
        if rod_info["number_of_holes"][-1] == 1:
            rod_type = "A"
        elif rod_info["number_of_holes"][-1] == 2:
            rod_type = "B"
        else:
            rod_type = "Unknown"

        rod_info["rod_type"].append(rod_type)
        print("Rod type: {}\n".format(rod_type))

        # Get the minimum enclosing rectangle of the rod
        box, (_, (width, height), angle) = find_MER(ext_contours[0])

        rod_info["length"].append(height)
        rod_info["width"].append(width)
        rod_info["angle"].append(angle)

        print("Rod Length: {:.4f}".format(rod_info["length"][-1]))
        print("Rod Width: {:.4f}".format(rod_info["width"][-1]))
        print("Rod Angle (deg): {:.4f}".format(rod_info["angle"][-1]))
        print("Rod Angle (rad): {:.4f}\n".format(np.deg2rad(rod_info["angle"][-1])))

        # Get the major and minor axis of the ellipse
        MA, ma = get_axes_by_ellipse(ext_contours[0])

        # Get the central moments of the rod, those are invariant to translation and scaling
        # To get invariance to scaling nu20, nu11, nu02, nu30, nu21, nu12, nu03 are used
        mu = get_moments(ext_contours)

        # Get the mass centers of the rod
        mc = get_mass_center(ext_contours, mu)
        rod_info["barycenter"].extend(mc)
        print("Rod mass center: {}".format(mc[0]))

        # Compute the width of the rod at the mass center
        width_mc, (p_left, p_right) = get_width_at_mass_center(binary_img, ext_contours[0], mu, mc[0])
        print("Width at mass center: {:.4f}".format(width_mc))

        # Get the position of the center of the hole(s)
        # We simply compute the mass center of the internal contours
        if n_holes > 0:
            holes_mc = get_mass_center(holes_contours, get_moments(holes_contours))
            rod_info["holes_barycenter"].append(holes_mc)
            print("Hole mass center: {}".format(holes_mc))

        # Obtain the diameters of the holes, we can simply assume that the holes are circles
        # and compute the diameter from the radius of the minimum enclosing circle
        diameter = get_holes_diameter(holes_contours)

        rod_info["holes_diameter"].append(diameter)    
        print("Hole diameter: {}".format(diameter))

        if visualize:

            # Draw the center mass
            m = mc[0]
            cv2.circle(output, (int(m[0]), int(m[1])), 4, (255, 0, 255), -1)
            cv2.putText(
                output,
                str(i),
                (int(m[0]) - 25, int(m[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )


            # Draw the MER
            cv2.drawContours(output, [box], -1, (0, 0, 255), 1)

            # Draw the contours
            cv2.drawContours(output, ext_contours, -1, (255, 0, 0), 1)
            cv2.drawContours(output, holes_contours, -1, (0, 255, 0), 1)

            # Draw the major and minor axes
            cv2.line(output, (MA[0], MA[1]), (MA[2], MA[3]), (255, 0, 255), 1)
            # cv2.line(output, (ma[0], ma[1]), (ma[2], ma[3]), (0, 0, 255), 1)

            cv2.circle(output, tuple(p_left), 2, (255, 0, 0), -1)       # red
            cv2.circle(output, tuple(p_right), 2, (0, 0, 255), -1)      # blue

            # # Draw the horizontal line passing through the barycenter
            # cv2.line(output, (0, m[1]), (img.shape[1], m[1]), (0, 255, 0), 1)
            
        print()
        print("*^" * 50)
        print()

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(output)
    ax.set_title("Rod {}".format(i))
    plt.show()

    return rod_info


def detect_rods(images: list, names: List[str], visualize=True):
    results = {}
    for image, name in zip(images, names):
        print("Processing image: {}".format(name))
        rod_info = detect_rods_blob(image, visualize=visualize)
        results[name] = rod_info

    return results
