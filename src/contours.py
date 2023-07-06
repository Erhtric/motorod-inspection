import cv2

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
        working_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
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
            # first_point = np.asarray(contours[j][0])
            # last_point = np.asarray(contours[j][-1])
            # print("First point: {}".format(first_point))
            # print("Last point: {}".format(last_point))
            # print("Distance: {}".format(euclidean_distance(first_point, last_point)))
            # If the point are not approximately the same, it is a broken contour

            holes_contours.append(contours[j])

    return contours, hierarchy, ext_contours, holes_contours