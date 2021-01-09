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
import cutOutPaintings_demo as cop
import pickle
import time

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

size = (1280,720) #size of the images we will compare
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



#db laden:
dataBase = {}
with open('database.bin', 'rb') as handle:
    dataBase = pickle.load(handle, encoding='latin1')

print(dataBase.keys())
#initializing keypoint creator
print("initializing keypoint creator..")
method = 'ORB'  # 'SIFT'
lowe_ratio = 0.89
if method   == 'ORB':
    finder = cv.ORB_create()
elif method == 'SIFT':
    finder = cv.xfeatures2d.SIFT_create()

font = cv.FONT_HERSHEY_SIMPLEX



cap = cv.VideoCapture(path_vids+'/'+sys.argv[1])

while(cap.isOpened()):
#    global wait
    ret, frame = cap.read()

    im = frame.copy()
    #im = cv.resize(im, (0,0), fx=0.5, fy=0.5)#helft hor en ver
    # calibrate
    #    print("\nim.shape[:-1]")
    #    print(im.shape[:-1])
    C_scale, roi = cv.getOptimalNewCameraMatrix(C_W, D_W, im.shape[:-1], 1, im.shape[:-1])
    #    print("C_scale")
    #    print(C_scale)
    #    print("C_M")
    #    print(C_M)
    #    print("\nRoi: ")
    #    print(roi)
    #calc map
    #mapx, mapy = cv.initUndistortRectifyMap(C_W, D_W,None, C_scale, im.shape[:-1], m1type = cv.CV_32FC1)
    # undistort
    im_rect = cv.undistort(im, C_W, D_W, None, C_W)
    #remap
    #im_rect = cv.remap(im, mapx, mapy, cv.INTER_LINEAR)
    #Python: cv2.remap(src, map1, map2, interpolation
    #mapx, mapy = cv2.initUndistortRectifyMap(intrinsic_matrix, distCoeff, None, newMat, size, m1type = cv2.CV_32FC1)

    cv.imshow('rectified stream',im_rect)
    k = cv.waitKey(wait) & 0xFF#wait 40ms for 25 FPS
    if(k == ord('s')):#s for save, q for quit
        time_label = strftime("%d %b %H-%M-%S", gmtime())
        print(time_label)
        cv.imwrite('/home/youssef/Documenten/Project computervisie/testImages/' + sys.argv[1] +'/rectified_calibM_'+str(time_label)+'.png', im_rect)
        #cv.imwrite('D:/School/2018-2019/Project CV - Paintings/Testimgs/rectified_calibM_'+str(time_label)+'.png', im_rect)


    ####################################################################################################################################
    ####################################################################################################################################
    ####################################################################################################################################
    if(k ==  ord('p')):#p to request a prediction
        paintings = cop.cut_out_paintings(im_rect)
        paintingsScaled = []
        inputDescriptors = []
        print(len(paintings))
        imageIndex = 0 #index of the part of the testing painting
        for img in paintings:
            #img = cv.resize(img, size)
            img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            paintingsScaled.append(img)
            kp2, des2 = finder.detectAndCompute(img,None)
            print(des2)

            #        print("\nLength of inputDescriptors:  ")
            #        print(len(inputDescriptors))
            topScores = {}

            imagesChecked = 0
            start = time.time()

            
            print("\nTEST: matchen in database...")
            print("inside inputDescriptors loop")
            #            	print("\ninputDesc: ")
            #            	print(inputDesc)
            for zaal, allDescriptors in dataBase.items(): #loop over alle zalen in de db
            # find the keypoints and descriptors with SIFT
                zaalBestScore = 0
                temp_totalMatchedKeyPoints= 0
                for descriptors in allDescriptors: #loop over alle schilderijen in die zaal
                #loop over alle schilderijen in de inputfoto
                #score = 0

                    imagesChecked += 1
                    # BFMatcher with default params
                    bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck = True)
                    #matches = bf.knnMatch(descriptors,inputDesc, k=2)
                    matches = bf.match(descriptors,des2)
                    totalMatchedKeyPoints = 0
                    print(len(matches))
                    for match in matches:
                        if match.distance < 150:
                            totalMatchedKeyPoints +=1
                    if temp_totalMatchedKeyPoints < totalMatchedKeyPoints:
                        temp_totalMatchedKeyPoints = totalMatchedKeyPoints

                #de score van de zaal wordt de score van de beste afbeelding in die map
                #topScores[imageIndex][zaal] = zaalBestScore
                topScores[zaal] = temp_totalMatchedKeyPoints
            done = time.time()
            elapsed = done-start
            print("\nTime to cut_out and compare to DB of keypoint and descriptors: ")
            print(round(elapsed,2))
            print("input image nr " + str(imageIndex))
                # first = 0
                # cv.putText(img,'score' + str(score), (10,100), font, 2,(255,255,255),2,cv.LINE_AA)
                # cv.putText(img, zaal, (10,45), font, 2,(255,255,255),2,cv.LINE_AA)
                # cv.imshow("image " + str(imageIndex),img)
                # cv.waitKey()
                # print(zaal + "    => " + str(score))
            print (topScores)
            cv.imshow("image " + str(imageIndex),img)
            cv.waitKey()
            imageIndex += 1
            print("")
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
