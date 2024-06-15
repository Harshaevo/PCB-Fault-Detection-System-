import numpy as np
import cv2
import imutils


#Returns the keypoints of ref_img and test_img that have higher similarity between each other.
def get_feat_keypoints(ref_img,test_img) :
    
    resize_ratio = 0.3
    #Calculate the new dimension of image
    new_x, new_y = [int(dim * resize_ratio) for dim in ref_img.shape]

    # Image downscale for accuracy improvement.Interpolation for reducing distortions 
    ref_img = cv2.resize(ref_img, (new_x, new_y), interpolation=cv2.INTER_LINEAR)
    test_img = cv2.resize(test_img, (new_x, new_y), interpolation=cv2.INTER_LINEAR)
    
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints(key points of the image) and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(ref_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(test_img, None)

    #brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) #with orb norm hamming distance is used as the measurement

    raw_matches = bf.knnMatch(descriptors1, descriptors2, 2) #knnmatch gives kth best match here it is k =2
    raw_matches = [matches for matches in raw_matches if len(matches) == 2]  # Keeps only pair of matches which has less distance
    matches = []

    # (Lowe's ratio test) ratio test to get best two matches
    for m, n in raw_matches:
        # ensure the distance is within a certain ratio of each
        if m.distance < n.distance * 0.75:
            matches.append(m)

    # Draw top matches
    imMatches = cv2.drawMatches(ref_img, keypoints1, test_img, keypoints2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("FaultImages\matches.jpg", imMatches)

    # Extract the coordinate pairs from good matches and resize them to original dimension
    src_pts = np.float32([[dim / resize_ratio for dim in keypoints1[m.queryIdx].pt] for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([[dim / resize_ratio for dim in keypoints2[m.trainIdx].pt] for m in matches]).reshape(-1, 1, 2)
    
    return src_pts, dst_pts


def align_image(ref_img, test_img):
    src_pts, dst_pts = get_feat_keypoints(ref_img, test_img)
    # Calculate Homography
    # Feature matching don't always find the best matches,so we need to use
    # a robust estimation technique called Random Sample Consensus (RANSAC)
    # that can calculate good results even with mediocre matches
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = ref_img.shape

    new_shape = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # Performs the perspective matrix transformation of vectors based on previously calculated homography 'M' and the new shape
    dst = cv2.perspectiveTransform(new_shape, M)

    # Calculates perspective transform from four pairs of the corresponding points gotten from previous calculation
    new_perspective = cv2.getPerspectiveTransform(np.float32(dst), new_shape)

    # Performs perspective transformation on target image
    alligned_image = cv2.warpPerspective(test_img, new_perspective, (w, h), borderValue=255)

    return alligned_image


def diff_and_morph(alligned_image, ref_img):
    kernel_dim = 3
    kernel_shape=None
    MORPHOLOGY_KERNELS = {'rect': cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_dim, kernel_dim)),
                          'ellipse': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dim, kernel_dim)),
                          'cross': cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_dim, kernel_dim))}

    difference = cv2.absdiff(alligned_image, ref_img)  # Absolute difference between both images

    #difference = remove_borders(difference)

    if kernel_shape in MORPHOLOGY_KERNELS:
        kernel = MORPHOLOGY_KERNELS[kernel_shape]
    else:
        kernel = MORPHOLOGY_KERNELS['ellipse']

    # Removes some imperfections due to potential jagged edges on image
    difference = cv2.erode(difference, kernel, iterations=3)  # erosion for removing small white noises (pixels near boundary are discarded)
    difference = cv2.dilate(difference, kernel, iterations=3)  # dilation for increasing the object (no noises here)

    return difference


def analyse_difference(ref_img, diff_image):
    # Finds the contours of white objects on diff_image
    cnts = cv2.findContours(diff_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    bounding_boxes = []

    has_defects = False

    for c in reversed(cnts):
        # Compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Copies a patch from both ref_img and diff_image corresponding
        # to the bounding boxes gotten from previous step
        patch1 = ref_img[y:y + h, x:x + w]
        patch2 = diff_image[y:y + h, x:x + w]

        # Measures how much the two HoG's are similar
        divergence = cv2.compareHist(np.float32(patch2), np.float32(patch1), cv2.HISTCMP_HELLINGER) #"Hellinger": cv2.HISTCMP_HELLINGER,

        # If divergence is above a threshold (depends on the method chosen in OPENCV_METHODS) then
        # append bounding box to list of defect spatial locations
        if divergence > 0.1:
            has_defects = True
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes, has_defects


def apply_bounding_boxes(img, bounding_boxes, color=(255, 255, 0)):
    delta = 5

    for bounding_box in bounding_boxes:
        (x, y, w, h) = bounding_box
        cv2.rectangle(img, (x - delta, y - delta), (x + w + delta, y + h + delta), color, 10)
    defected = cv2.copyMakeBorder(src=img, top=5, bottom=5, left=5, right=5, borderType=cv2.BORDER_CONSTANT) 
    
    return defected


def bgremove(myimage):
    # Create a white background with the same dimensions as the image
    white_background = np.ones_like(myimage) * 255  # White is represented as (255, 255, 255) in BGR

    # Define the color of the bounding boxes (e.g., red)
    box_color = (0,0,0)  # BGR color (here, red)

    # Find all the pixels in the image that match the bounding box color
    matching_pixels = np.all(myimage == box_color, axis=2)

    # Set all matching pixels in the white background to the same color
    white_background[matching_pixels] = box_color

    filename = f"FaultImages\ColourFiltered.jpg"
    cv2.imwrite(filename, white_background)

    return filename


def main(img1,img2):

    ref_img = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)

    test_img = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    defected_img = cv2.imread(img2)

    _, ref_img = cv2.threshold(ref_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  #Otsu thresholding->Optimal threshold value
    _, test_img = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 

    aligned_image = align_image(ref_img, test_img)

    difference = diff_and_morph(aligned_image, ref_img)

    bounding_boxes, has_defects = analyse_difference(ref_img, difference)

    defects = apply_bounding_boxes(defected_img, bounding_boxes,False)

    print(f'This PCB is a faulty one? {has_defects}')

    filename01 = f"FaultImages\DefectDetected.jpg"
    cv2.imwrite(filename01, defects)

    filename02 = bgremove(defects)

    return filename01,filename02