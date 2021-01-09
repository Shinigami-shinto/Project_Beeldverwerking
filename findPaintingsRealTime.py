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
import utils
#np.set_printoptions(threshold=np.nan)

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
path_vids = "/home/youssef/Documenten/Project computervisie" #MSK_01.mp4
count = 1

#for camera calibration
dims = (6,10)
objp = np.zeros((dims[0]*dims[1],3), np.float32)
objp[:,:2] = np.mgrid[0:dims[0],0:dims[1]].T.reshape(-1,2)

C = np.array([[ 5.6729034524746328e+02, 0., 6.3764777940570559e+02], [0.,
       5.7207768469558505e+02, 3.3299427011674493e+02], [0., 0., 1. ]])
D = np.array([ -2.4637408439446815e-01, 7.6662428015464898e-02,
       -2.7014001885212116e-05, -3.1925229062179259e-04,
       -1.2400436109816003e-02 ])



##############################################################################
#####    MAIN     ############################################################
##############################################################################
cap = cv2.VideoCapture(path_vids+'/'+sys.argv[1])

while(cap.isOpened()):
    ret, frame = cap.read()
    
    im = frame.copy()
    # calibrate
    C_scale, roi = cv2.getOptimalNewCameraMatrix(C, D, im.shape[:-1], 1, im.shape[:-1])

    # undistort
    im_rect = cv2.undistort(im, C, D, None, C)
    
    # cv2.imshow('rectified stream',im_rect)
    found_stuff = utils.real_time(im_rect)
    cv2.imshow('rectified stream',found_stuff)
    k = cv2.waitKey(10) & 0xFF#wait 40ms for 25 FPS
    if(k == ord('s')):#s for save, q for quit
        cv2.imwrite('D:/School/2018-2019/Project CV - Paintings/Testimgs/rect_img'+str(count)+'.png', im_rect)
        count+=1
    elif(k == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()