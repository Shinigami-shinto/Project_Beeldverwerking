# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:23:47 2019

@author: Felix De Mûelenaere
"""
##############################################################################
########################### Generating test images ###########################
#####                        from MSK gopro videos                        ####
##############################################################################
#Imports
import cv2
import sys
import numpy as np
np.set_printoptions(threshold=np.nan)

if(not sys.argv[1]):
        print("No imagename given, please provide it as the first argument at "
              + "the CLI!!")
        sys.exit();

##############################################################################
#####    things to think about    ############################################
##############################################################################
'''
1. Apply pre-processing to get descent test images (see minerva about
camera corection and calibration)
2. Use waitKey to obtain 25 FPS, press a certain key to save the test-img
'''
##############################################################################
#####    Variables & Functions    ############################################
##############################################################################
path_vids = "D:/School/2018-2019/Project CV - Paintings" #MSK_01.mp4
count = 1


##############################################################################
#####    MAIN     ############################################################
##############################################################################
cap = cv2.VideoCapture(path_vids+'/'+sys.argv[1])

while(cap.isOpened()):
    ret, frame = cap.read()

#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(40) & 0xFF#wait 40ms for 25 FPS
    if(k == ord('s')):#s for save, q for quit
        cv2.imwrite('D:/School/2018-2019/Project CV - Paintings/Testimgs/'+str(count)+'.png', frame)
    elif(k == ord('q')):
        break
    count+=1

cap.release()
cv2.destroyAllWindows()