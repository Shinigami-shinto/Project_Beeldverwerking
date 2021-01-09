# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:23:47 2019

@author: Felix De Mûelenaere
"""
##############################################################################
########################### Generating test images ###########################
#####                        from MSK gopro videos                        ####
#####                     & DEMO mode                                     ####
##############################################################################
#Imports
import cv2 as cv
import sys
import numpy as np
from time import gmtime, strftime
import utils
import pickle
import time
from multiprocessing.pool import ThreadPool

if(not sys.argv[1]):
        print("No imagename given, please provide it as the first argument at "
              + "the CLI!!")
        sys.exit();

##############################################################################
#####    How to use               ############################################
##############################################################################
'''
press 'p' to get a prediction of what room you are in
press 'q' to quit
press 's' to save
press 'x' to speed up FPS
press 'y' to slow down FPS

'''
##############################################################################
#####    Functions                ############################################
##############################################################################
'''

'''
##############################################################################
#####    Variables & Functions    ############################################
##############################################################################
#path_vids = "D:/School/2018-2019/Project CV - Paintings" #MSK_01.mp4
path_vids = "/home/youssef/Documenten/Projectcomputervisie"

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

wait = 66#to determine the framerate of the video
##############################################################################
#####    MAIN     ############################################################
##############################################################################

#opencl aanzetten:
print("opencl?")
print("uses opencl first: " + str(cv.ocl.useOpenCL()))
print("has opencl: " + str(cv.ocl.haveOpenCL()))
cv.ocl.setUseOpenCL(True)
print("uses opencl: " + str(cv.ocl.useOpenCL()))

font = cv.FONT_HERSHEY_SIMPLEX

cap = cv.VideoCapture(path_vids+'/'+sys.argv[1])
pt = ThreadPool(6)
utils.initialize_database()

while(cap.isOpened()):
#    global wait
    ret, frame = cap.read()
    im = frame.copy()
    C_scale, roi = cv.getOptimalNewCameraMatrix(C_W, D_W, im.shape[:-1], 1, im.shape[:-1])
    im_rect = cv.undistort(im, C_W, D_W, None, C_W)
    cv.imshow('rectified stream',im_rect)
    k = cv.waitKey(wait) & 0xFF#wait 40ms for 25 FPS
    if(k == ord('s')):#s for save, q for quit
        time_label = strftime("%d %b %H-%M-%S", gmtime())
        print(time_label)
        cv.imwrite('/home/youssef/Documenten/Projectcomputervisie/testImages/' + sys.argv[1] +'/rectified_calibM_'+str(time_label)+'.png', im_rect)
        #cv.imwrite('D:/School/2018-2019/Project CV - Paintings/Testimgs/rectified_calibM_'+str(time_label)+'.png', im_rect)
        
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
    if(k ==  ord('p')):#p to request a prediction
        img = im_rect
        if not utils.is_too_blurry(img):
            print("Trying to find possible paintings...")
            possiblePaintings = utils.cut_out_paintings(img)
            if len(possiblePaintings)!=0:
                descPerPaitings = pt.map(utils.get_desc_and_keypoints,possiblePaintings)
                print("Matching with db...")
                scores = pt.map(utils.match_with_db, descPerPaitings)
                predictionPerPainting = []
                for topScoresImage in scores:
                    if len(topScoresImage) != 0:
                        #print("input image nr " + str(imageIndex))
                        topscoresSorted = sorted(topScoresImage.items(), key=lambda x: -x[1])
                        score = topscoresSorted[0][1]
                        besteZaal = topscoresSorted[0][0]
                        if score == 0:
                            besteZaal = "geen painting"
                        predictionPerPainting.append((besteZaal,score))
                    else:
                        predictionPerPainting.append(("geen painting",0))
                if(len(predictionPerPainting) != 0):
                    predictionPerPaintingSorted = sorted(predictionPerPainting, key=lambda x: -x[1])
                    #print(predictionPerPaintingSorted)
                    predictionPerPainting = predictionPerPaintingSorted[0]
                cv.putText(img,predictionPerPainting[0],(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
                cv.imshow("Prediction",img)
                cv.waitKey(0)
                print("We predicted:", predictionPerPainting)
            else:
                print("No possible paintings found")
        else:
            print("Image is too blurry")
    elif(k == ord('q')):#q to quit
        break
    elif(k == ord('x')):#x to speed up FPS
        wait = int(wait/1.2)
        print("FPS is now:        "+str(1000/wait))
    elif(k == ord('y')):#x to slow down FPS
        wait = int(wait*1.2)
        print("FPS is now:        "+str(1000/wait))
        

cap.release()
cv.destroyAllWindows()
