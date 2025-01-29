import cv2
from os import path
import numpy as np
import glob
import matplotlib.pyplot as plt


def find_coin(image, show=False):
    dirname = path.dirname(__file__)
    coin_patterns_dir = path.join(dirname, "coin_patterns" , "*.jpg")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
   
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT_ALT, 
        dp=1.0, 
        minDist=300,
        param1=200, 
        param2=0.9, 
        minRadius=50, 
        maxRadius=200 
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        patterns = []
        for pattern_path in glob.glob(coin_patterns_dir):
            img = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(img, None)
            patterns.append((img, keypoints, descriptors))

        for circle in circles[0, :]:
            x, y, r = circle
            circle_roi = image[y-r:y+r, x-r:x+r]

            # Wykrywanie punktów charakterystycznych
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(circle_roi, None)
            
            best_matches = 0
            for pattern_img, kp_pattern, des_pattern in patterns:
                
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des_pattern, descriptors)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > best_matches:
                    best_matches = len(matches)

            if best_matches > 60:
                if show:
                    plt.imshow(cv2.cvtColor(circle_roi , cv2.COLOR_BGR2RGB))
                return (x,y,r)
                # cv2.circle(image, (x, y), r, (0, 255, 0), 2)

    return (0,0,0)

    imgS = cv2.resize(image,(600,800))
    cv2.imshow("Result", imgS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
