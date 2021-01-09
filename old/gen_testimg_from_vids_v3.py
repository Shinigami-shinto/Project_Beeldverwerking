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
from time import gmtime, strftime, sleep
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
#####    Functions                ############################################
##############################################################################
'''
#Python: cv2.undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]
The function transforms an image to compensate radial and tangential lens distortion.

The function is simply a combination of initUndistortRectifyMap() (with unity R ) and remap() (with bilinear interpolation).
'''
##############################################################################
#####    Variables & Functions    ############################################
##############################################################################
path_vids = "D:/School/2018-2019/Project CV - Paintings" #MSK_01.mp4

#for camera calibration
'''calib_W'''
C_W = np.array([[ 5.6729034524746328e+02, 0., 6.3764777940570559e+02], [0.,
       5.7207768469558505e+02, 3.3299427011674493e+02], [0., 0., 1. ]])
D_W = np.array([ -2.4637408439446815e-01, 7.6662428015464898e-02,
       -2.7014001885212116e-05, -3.1925229062179259e-04,
       -1.2400436109816003e-02 ])

'''calib_M'''
C_M = np.array([[7.2337882890945207e+02, 0., 6.4226033453805235e+02], [0.,
       7.2844995950341502e+02, 3.2297129949442024e+02], [0., 0., 1.]])
D_M = np.array([-2.7971075073202351e-01, 1.2737835217024596e-01,
       5.5264049900636148e-04, -2.4709811526299534e-04,
       -3.7787805887358195e-02])


##############################################################################
#####    MAIN     ############################################################
##############################################################################
cap = cv2.VideoCapture(path_vids+'/'+sys.argv[1])
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
#    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if(frame is not None):
        im = frame.copy()
        im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)#helft hor en ver
        
        if(cap.get(cv2.CAP_PROP_POS_FRAMES)%100==0):
            print("\n100 frames have passed")
            C_scale, roi = cv2.getOptimalNewCameraMatrix(C_M, D_M, im.shape[:-1], alpha = 0.5)
            #calc map
            mapx, mapy = cv2.initUndistortRectifyMap(C_M, D_M,None, C_scale, im.shape[:-1], m1type = cv2.CV_32FC1)
            #remap
            im_rect = cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR)
            cv2.imwrite('D:/School/2018-2019/Project CV - Paintings/Testimgs/rectified_calibM_v2_s'+str(count)+'.png', im_rect)
            count += 1

cap.release()
cv2.destroyAllWindows()