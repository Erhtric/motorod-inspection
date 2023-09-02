import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import List
from src.preprocess import apply_gaussian_filter, binarize
from src.blob import get_axes_by_ellipse, get_width_at_mass_center, get_connected_component, get_moments, get_mass_center, find_MER, get_holes_diameter
from src.contours import find_contours

logging.basicConfig(level=logging.INFO)

def detect_contact_points(img, min_area):
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
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contact_points = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            # Approximate the contour with a polygon
            # https://docs.opencv.org/3.4/d7/d1d/tutorial_hull.html
            contour = cv2.approxPolyDP(contour, 3, True)
            contoursHull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, contoursHull)

            for j in range(defects.shape[0]):
                defpoint = defects[j][0]
                pt = tuple(contour[defpoint[2]][0])  # Get defect point
                rect = (pt[0] - 2, pt[1] - 2, 5, 5)  # Create 5x5 Rect from defect point

                non_zero_pixels = np.count_nonzero(img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
                if non_zero_pixels > 18:
                    contact_points.append(pt)

    return contact_points

def separate_rods(img, contact_points, distance, thickness):
    """
    This method separates the rods in the given image given a list of contact points. We firstly 
    detect those contact points and then we draw a black line between the nearby contact points. 

    Parameters
    ----------
    img : numpy.ndarray
        Binary Image to process.
    contact_points : list
        List of contact points.
    distance : int, optional
        Distance between contact points to consider
        
    Returns
    -------
    img : numpy.ndarray
        Binary Image with the rods separated.
    """
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img = cv2.dilate(img, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    img = cv2.erode(img, kernel, iterations=2)

    for i in range(len(contact_points)):
        for j in range(i + 1, len(contact_points)):
            if np.linalg.norm(np.array(contact_points[i]) - np.array(contact_points[j])) < distance:
                # Draw a black line between the contact points
                cv2.line(img, contact_points[i], contact_points[j], 0, thickness)

    return img

def filter_rods_by_area(rod_info, labels, num_labels, stats, min_area):
    # Filter out in the labels the small components
    rod_info["labels"] = np.zeros_like(labels)
    count = 1
    for i in range(1, num_labels):          # Skip background
        if stats[i][4] > min_area:
            rod_info["labels"][labels == i] = count
            count += 1

    stats = stats[stats[:, 4] > min_area]
    
    rod_info["stats"] = stats
    rod_info["num_labels"] = rod_info["stats"].shape[0]
    rod_info["number_of_rods"] = rod_info["num_labels"] - 1  # Remove background

    return rod_info, stats, labels, num_labels

def detect_rods_blob(image, min_area=None, detect_contact_pts=False, visualize=True, name=""):
    """
    This method detects the rods objects in the given image using blob detection and
    counts the number of holes in each rod.

    Parameters
    ----------
    img : numpy.ndarray
        Image to process.
    min_area : int, optional
        Minimum area of the blobs to consider, by default None
    visualize : bool, optional
        Whether to visualize the results, by default True

    Returns
    -------
    dict
        Dictionary containing information about the rods.
    """
    img = image.copy()
    output = image.copy()       # This serves only for visualization purposes
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    # Dictionary containing information about the rods
    rod_info = {
        "num_labels": None,
        "labels": None,
        "number_of_rods": None,
        "stats": None,
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

    if detect_contact_pts and min_area is not None:
        contact_points = detect_contact_points(binary_img, min_area)
        for c in contact_points:
            cv2.circle(output, c, 2, (0, 255, 255), -1)

        binary_img = separate_rods(binary_img, contact_points, distance=20, thickness=2)

    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gae57b028a2b2ca327227c2399a9d53241
    # Find the connected components in the binary image
    num_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(binary_img, connectivity=4)

    if min_area is not None:
        logging.info("Filtering rods by area...")
        rod_info, stats, labels, num_labels = filter_rods_by_area(rod_info, labels, num_labels, stats, min_area)
    else:
        rod_info["number_of_rods"] = num_labels - 1  # Remove background
        rod_info["num_labels"] = num_labels
        rod_info["labels"] = labels

    logging.info("Number of rods found (CC): {}".format(rod_info["number_of_rods"]))

    for i in tqdm(range(1, rod_info["num_labels"]), desc="Processing rods"):
        # Loop over the connected components, 0 is the background
        logging.info("\nProcessing rod {}...".format(i))
        # Get the masked image, now the image will contain only one rod
        comp = get_connected_component(rod_info["labels"], i)

        _, _, _, _, area = stats[i]
        rod_info["area"].append(area)

        logging.info("Centroid (CC): {}".format(tuple(np.int0(centroid[i]))))
        logging.info("Area (CC): {}".format(area))

        contours, hierarchy, ext_contours, holes_contours = find_contours(comp)
        rod_info["contours"].append(contours)
        rod_info["hierarchy"].append(hierarchy)

        # A hole has a parent contour but no child contour
        n_holes = sum(
            1
            for j in range(hierarchy[0].shape[0])
            if hierarchy[0][j][2] == -1 and hierarchy[0][j][3] > -1
        )

        logging.info("Number of holes: {}".format(n_holes))
        rod_info["number_of_holes"].append(n_holes)

        # Rod type detection
        if rod_info["number_of_holes"][-1] == 1:
            rod_type = "A"
        elif rod_info["number_of_holes"][-1] == 2:
            rod_type = "B"
        else:
            rod_type = "Unknown"

        rod_info["rod_type"].append(rod_type)
        logging.info("Rod type: {}".format(rod_type))

        # Get the minimum enclosing rectangle of the rod
        box, (_, (width, height), angle) = find_MER(ext_contours[0])

        rod_info["length"].append(height)
        rod_info["width"].append(width)
        rod_info["angle"].append(angle)

        logging.info("Rod Length: {:.2f}".format(rod_info["length"][-1]))
        logging.info("Rod Width: {:.2f}".format(rod_info["width"][-1]))
        logging.info("Rod Angle (deg): {:.2f}°".format(rod_info["angle"][-1]))

        # Get the major and minor axis of the ellipse
        MA, ma = get_axes_by_ellipse(ext_contours[0])

        # Get the central moments of the rod, those are invariant to translation and scaling
        # To get invariance to scaling nu20, nu11, nu02, nu30, nu21, nu12, nu03 are used
        mu = get_moments(ext_contours)

        # Get the mass centers of the rod
        mc = get_mass_center(ext_contours, mu)
        rod_info["barycenter"].extend(mc)
        logging.info("Rod mass center: {}".format(mc[0]))

        # Compute the width of the rod at the mass center
        width_mc, (p_left, p_right) = get_width_at_mass_center(binary_img, ext_contours[0], mu, mc[0])
        logging.info("Width at mass center: {:.2f}".format(width_mc))

        # Get the position of the center of the hole(s)
        # We simply compute the mass center of the internal contours
        if n_holes > 0:
            holes_mc = get_mass_center(holes_contours, get_moments(holes_contours))
            rod_info["holes_barycenter"].append(holes_mc)
            logging.info("Hole mass center: {}".format(holes_mc))

        # Obtain the diameters of the holes, we can simply assume that the holes are circles
        # and compute the diameter from the radius of the minimum enclosing circle
        diameter = get_holes_diameter(holes_contours)

        rod_info["holes_diameter"].append(diameter)    
        logging.info("Hole diameter: {}\n".format(diameter))

        if visualize:
            # Draw the center mass
            m = mc[0]
            cv2.circle(output, (int(m[0]), int(m[1])), 3, (255, 0, 255), -1)
            cv2.putText(
                output,
                str(i),
                (int(m[0]) - 20, int(m[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

            # Draw the center of the hole(s)
            if n_holes > 0:
                for h_i, hole in enumerate(holes_mc):
                    cv2.circle(output, (int(hole[0]), int(hole[1])), 3, (0, 150, 0), -1)

                    # Draw the number of the hole
                    cv2.putText(
                        output,
                        str(h_i + 1),
                        (int(hole[0]) - 3, int(hole[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 150, 0),
                        1,
                    )

            # Draw the MER
            cv2.drawContours(output, [box], -1, (100, 0, 255), 1)

            # Draw the contours
            cv2.drawContours(output, ext_contours, -1, (200, 100, 0), 1)
            cv2.drawContours(output, holes_contours, -1, (0, 150, 0), 1)

            # Draw the major and minor axes
            cv2.line(output, (MA[0], MA[1]), (MA[2], MA[3]), (255, 0, 255), 1)
            # cv2.line(output, (ma[0], ma[1]), (ma[2], ma[3]), (0, 0, 255), 1)

            cv2.circle(output, tuple(p_left), 2, (255, 0, 0), -1)       # red
            cv2.circle(output, tuple(p_right), 2, (0, 0, 255), -1)      # blue

            # # Draw the horizontal line passing through the barycenter
            # cv2.line(output, (0, m[1]), (img.shape[1], m[1]), (0, 255, 0), 1)
            
        logging.info("*^" * 50 + "\n")

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(output)
        ax.set_title(name)
        plt.show()
        
    if name != "": 
        logging.info("Saving image: {}".format(name))
        plt.savefig("./images/" + name.replace(".BMP", "") + ".png")


    return rod_info