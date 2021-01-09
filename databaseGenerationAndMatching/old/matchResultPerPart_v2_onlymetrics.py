import argparse
import os
import numpy as np
import cv2 as cv
import cutOutPaintingsDatabase as cop
import pickle
import time

size = 750 #size of the images we will compare
defaultSource = 'database.bin'

path_selfgen_test = "/home/youssef/Documenten/Projectcomputervisie/testImages/testSetPerZaal"
path_single_paintings_abstract = "D:/School/2018-2019/Project CV - Paintings/RAW dataset CV_AJ1819/pictures_msk_smak_galaxy_A5/single_paintings/houses_villages/"


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False,
	help="path to the binary file (database)")
args = vars(ap.parse_args())
if not args["source"]:
	args["source"]= defaultSource
srcFile = args["source"];
print(srcFile);

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
    
dataBaseTest = {}
with open('databaseTest.bin', 'rb') as handle:
    dataBaseTest = pickle.load(handle, encoding='latin1')


#initializing keypoint creator
print("initializing keypoint creator..")
method = 'ORB'  # 'SIFT'
lowe_ratio = 0.89
if method   == 'ORB':
    finder = cv.ORB_create()
elif method == 'SIFT':
    finder = cv.xfeatures2d.SIFT_create()
    
font = cv.FONT_HERSHEY_SIMPLEX

#metrics
TP=FP=TN=FN=cutOutDidntWork=0

def TESTING():#gaat alle testafbeeldingen af en geeft telkens alle votes
    global TP,FP,TN,FN,cutOutDidntWork
    for dirname, dirnames, filenames in os.walk(path_selfgen_test):
        for f in sorted(filenames, key=len): #alle fotos van de zalen overlopen
            if f.startswith('.'):
                continue
#            print("               YOHOHOHOHO filenames:      " + f)
            print(f)
            dirname += "/"
            start = time.time()
#            print(dirname+f)
#            src = cv.imread(dirname+f)
#            cv.imshow("troubleshooting",src)
#            cv.waitKey()
            paintings = cop.cut_out_paintings(f , dirname)
            paintingsScaled = []
            inputDescriptors = []
            for img in paintings:
                #img = cv.resize(img, (size, size))
                img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                #cv.imshow("cut",img)
                #cv.waitKey()
                paintingsScaled.append(img)
                kp2, des2 = finder.detectAndCompute(img,None)
                
                inputDescriptors.append(des2)
                
    #        print("\nLength of inputDescriptors:  ")
    #        print(len(inputDescriptors))
            if len(paintings)==0: cutOutDidntWork+=1
            topScores = []
            
            imagesChecked = 0
            
            imageIndex = 0 #index of the part of the testing painting
            for inputDesc in inputDescriptors: 
                print("\nTEST: matchen in database...")
                print("inside inputDescriptors loop")
                if(inputDesc is not None):
    #            	print("\ninputDesc: ")
    #            	print(inputDesc)
                	topScores.append({})
                	for zaal, allDescriptors in dataBase.items(): #loop over alle zalen in de db
                	# find the keypoints and descriptors with SIFT
                		zaalBestScore = 0
                		for descriptors in allDescriptors: #loop over alle schilderijen in die zaal
                		#loop over alle schilderijen in de inputfoto
                			score = 0
                			imagesChecked += 1;
                			# BFMatcher with default params
                			bf = cv.BFMatcher()
                			matches = bf.knnMatch(descriptors,inputDesc, k=2)
    #            			print("len matches[0]")
    #            			print(len(matches[0]))
                			
                			# Apply ratio test
                			good = []
                			if(len(matches[0]) > 1):
                				for m,n in matches:
                				    if m.distance < lowe_ratio*n.distance:
                				        good.append([m])
                				        score += 10.0 / m.distance;
                				if score > zaalBestScore:
                					zaalBestScore = score;
                			
                		#de score van de zaal wordt de score van de beste afbeelding in die map
                		topScores[imageIndex][zaal] = zaalBestScore
                	imageIndex += 1;
        
        
        
                
                done = time.time()
                elapsed = done-start
                print("\nTime to cut_out and compare to DB of keypoint and descriptors: ")
                print(round(elapsed,2))
                imageIndex = 0
                for topScoresImage in topScores:
                
                	print("input image nr " + str(imageIndex))
                	first = 1
                	for zaal, score in sorted(topScoresImage.items(), key=lambda x: -x[1]):
                		if score > 3.6:
                			if first == 1:
                				first = 0
#                				cv.putText(paintingsScaled[imageIndex],'score' + str(score), (10,100), font, 2,(255,255,255),2,cv.LINE_AA)
#                				cv.putText(paintingsScaled[imageIndex], zaal, (10,45), font, 2,(255,255,255),2,cv.LINE_AA)
#                				cv.imshow("image " + str(imageIndex), paintingsScaled[imageIndex])
#                				cv.waitKey()
                				if(dataBaseTest[f] == zaal):
                					TP += 1
                				elif(dataBaseTest[f] != zaal):
                					FP += 1
                			print(zaal + "    => " + str(score))
                		elif score < 3.6:
                			if first == 1:
                				first = 0
#                				cv.putText(paintingsScaled[imageIndex],'UNKNOWN', (10,100), font, 2,(255,255,255),2,cv.LINE_AA)
#                				cv.imshow("unknown" + str(imageIndex), paintingsScaled[imageIndex])
                				print(zaal + "    => " + str(score))
#                				cv.waitKey()
                				if(dataBaseTest[f] == zaal):
                					FN += 1
                				elif(dataBaseTest[f] != zaal):
                					TN += 1
                
                	print("")
                	imageIndex += 1;

TESTING()

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP+TN+FP+FN)
print("\n\nPrecision:            " + str(precision))
print("\nRecall:            " + str(recall))
print("\nAccuracy:            " + str(accuracy))
print("\nNumber of images where cutOut didn't find anything: ",cutOutDidntWork)

cv.waitKey()
cv.destroyAllWindows()