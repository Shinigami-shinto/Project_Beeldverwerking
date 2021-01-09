import cv2
import numpy as np
import matplotlib.pyplot as plt

import cutOutPaintings as cop

path = './RAW_DATASET/zaal_17/IMG_20190323_120515.jpg'
path2= './RAW_DATASET/zaal_17/testimg7.jpg'


#paintings is een array van afbeelding (alle schilderijen uitgesneden uit de afbeelding) 
paintings = cop.cut_out_paintings(path)#van RAW dataset
paintings2= cop.cut_out_paintings(path2)#test-afbeelding genomen uit goPro video

#count=1
#for painting in paintings2:
#    cv2.imshow('painting2.index: '+str(count), painting)
#    count+=1

gray  = cv2.cvtColor(paintings[1], cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(paintings2[0], cv2.COLOR_BGR2GRAY)

cv2.imshow('gray',gray)
cv2.imshow('gray2',gray2)

#'''

# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=50)
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(gray,None)
kp2, des2 = orb.detectAndCompute(gray2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(gray,kp1,gray2,kp2,matches[:10],None, flags=2) #display first 10 matches

plt.imshow(img3),plt.show()
#'''

cv2.waitKey()
cv2.destroyAllWindows()