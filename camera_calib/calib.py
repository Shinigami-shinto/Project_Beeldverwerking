# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

dims = (6,10)
objp = np.zeros((dims[0]*dims[1],3), np.float32)
objp[:,:2] = np.mgrid[0:dims[0],0:dims[1]].T.reshape(-1,2)

im = cv2.imread("../../frames/W/calib_M_00074.png", cv2.IMREAD_COLOR)

#C = np.array([[7.2337882890945207e+02, 0., 6.4226033453805235e+02], [0.,
#       7.2844995950341502e+02, 3.2297129949442024e+02], [0., 0., 1.]])
#D = np.array([-2.7971075073202351e-01, 1.2737835217024596e-01,
#       5.5264049900636148e-04, -2.4709811526299534e-04,
#       -3.7787805887358195e-02])

C = np.array([[ 5.6729034524746328e+02, 0., 6.3764777940570559e+02], [0.,
       5.7207768469558505e+02, 3.3299427011674493e+02], [0., 0., 1. ]])
D = np.array([ -2.4637408439446815e-01, 7.6662428015464898e-02,
       -2.7014001885212116e-05, -3.1925229062179259e-04,
       -1.2400436109816003e-02 ])

# calibrate
#C = np.zeros((3, 3), dtype=float)
#D = np.zeros((4,), dtype=float)
#ret, C, D, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], im.shape[:-1], C, D)
C_scale, roi = cv2.getOptimalNewCameraMatrix(C, D, im.shape[:-1], 1, im.shape[:-1])

# undistort
im_rect = cv2.undistort(im, C, D, None, C)
cv2.imwrite('rectified.jpg',im_rect)

