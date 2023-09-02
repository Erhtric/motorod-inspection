# Visual Inspection of Motorcycle Connecting Rods using OpenCV.

This project uses OpenCV to perform visual inspection of motorcycle connecting rods. The goal of the project is to detect and analyze connecting rods in images, and to identify any defects or abnormalities in the rods.

The project is divided into two main tasks:

- **First Task**:

   The goal of this section is to detect and analyze connecting rods in images and to provide information about the type of rod, position and orientation, length, width, and width at the barycenter, and position and diameter of each hole. The `detect_rods_blob` function is described, which takes an image and a dictionary containing the salient information about the rods as input. The function performs the following steps:

    1. Applies a Gaussian filter to the image to reduce noise.
    2. Binarizes the image through a thresholding operation.
    3. Detects the different objects using the `connectedComponentsWithStats` method, which computes the connected components labeled of a binary image.
    4. For each connected component, the function finds the contours via the `findContours` method and identifies the type of rod by counting the number of holes.
    5. Computes the minimum enclosing rectangle of the external contours by using the `minAreaRect` function to obtain the length, width, and angle of the rod.
    6. Computes the width at the barycenter of the rod
    7. Computes the position and diameter of each hole by using the `minEnclosingCircle` function.
        
- **Second Task**:

    The goal of this section is to detect and analyze connecting rods in images while addressing additional challenges such as distractors, iron powder, and contact points.
    The section describes the specific challenges and how they were addressed:

    1. **Distractors**: The `cv2.connectedComponentsWithStats` method is used to find the connected components of the image. The area of the connected components is then used to filter out the distractors by setting a threshold on the area of the connected components. This is done by passing a `min_area` parameter to the `detect_rods_blob` function.
    2. **Iron powder**: The `min_area` parameter is used to filter out the iron powder from the image by setting a threshold on the area of the connected components. The best value for `min_area` was found to be `1500`.
    3. **Contact points**: The task is separated into two main focal points: contact points detection and rods separation. For contact points detection, the code iterates over every external contour, approximates it, and computes the convex hull on the approximated contour. The convexity defects of the contours are then found using `cv2.convexityDefects`. For each defect found, the code counts the non-zero pixels in a small patch around the point. If the amount of non-zero pixels is greater than a threshold (set to `18`), the point is considered a contact point. For rods separation, the code performs a dilation operation with a $2\times2$ kernel and an erosion operation with a $2\times1$ kernel. The code then iterates over each pair of contact points and discards the ones that are too far away from each other. A black line is then drawn between the two points to separate the two rods in the binary image.