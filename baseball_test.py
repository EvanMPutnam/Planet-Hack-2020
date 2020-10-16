import cv2
import numpy as np
from matplotlib import pyplot as plt


#STARTING_IMG = 'baseball2.PNG'
STARTING_IMG = 'bbf.png'
#OTHER_IMG = 'baseball3.PNG'
OTHER_IMG = 'bbf_in_scene.png'

def test1():
    img1 = cv2.imread(STARTING_IMG, 0)
    img2 = cv2.imread(OTHER_IMG)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # draw first 50 matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    cv2.imshow('Matches', match_img)
    cv2.waitKey()

def test2():

    print ("Loading")
    img1 = cv2.imread(STARTING_IMG,0)    # queryImage
    img2 = cv2.imread(OTHER_IMG,0) # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    print ("Loading")

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    print ("Loading")

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)

    plt.imshow(img3)
    plt.show()
    print ("Done")

test2()