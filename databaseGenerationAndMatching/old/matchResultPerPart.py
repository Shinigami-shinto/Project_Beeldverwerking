import argparse
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import cutOutPaintings as cop
import pickle

size = 750 #size of the images we will compare
defaultSource = 'database.bin'
testImage = "5.jpg"


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False,
	help="path to the binary file (database)")
args = vars(ap.parse_args())
if not args["source"]:
	args["source"]= defaultSource
srcFile = args["source"];
print(srcFile);

#opencl aanzetten:
print "opencl?"
print("uses opencl first: " + str(cv.ocl.useOpenCL()))
print("has opencl: " + str(cv.ocl.haveOpenCL()))
cv.ocl.setUseOpenCL(True)
print("uses opencl: " + str(cv.ocl.useOpenCL()))



#db laden:
dataBase = {};
with open('database.bin', 'rb') as handle:
    dataBase = pickle.load(handle)


#initializing keypoint creator
print "initializing keypoint creator.."
method = 'ORB'  # 'SIFT'
lowe_ratio = 0.89
if method   == 'ORB':
    finder = cv.ORB_create(nfeatures=500)
elif method == 'SIFT':
    finder = cv.xfeatures2d.SIFT_create()

print "TEST: segmenten uit afbeelding halen..."

paintings = cop.cut_out_paintings(testImage)
paintingsScaled = []
inputDescriptors = []
for img in paintings:
	img = cv.resize(img, (size, size))
	paintingsScaled.append(img)
	kp2, des2 = finder.detectAndCompute(img,None)

	inputDescriptors.append(des2)

print "TEST: matchen in database..."
topScores = []

imagesChecked = 0

imageIndex = 0 #index of the part of the testing painting
for inputDesc in inputDescriptors: 
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

			# Apply ratio test
			good = []

			for m,n in matches:
			    if m.distance < lowe_ratio*n.distance:
			        good.append([m])
			        score += 10.0 / m.distance;
			if score > zaalBestScore:
				zaalBestScore = score;
			
		#de score van de zaal wordt de score van de beste afbeelding in die map
		topScores[imageIndex][zaal] = zaalBestScore
	imageIndex += 1;

font = cv.FONT_HERSHEY_SIMPLEX


imageIndex = 0
for topScoresImage in topScores:

	print "input image nr " + str(imageIndex)	
	first = 1
	for zaal, score in sorted(topScoresImage.items(), key=lambda x: -x[1]):
		if score > 0:
			if first == 1:
				first = 0
				cv.putText(paintingsScaled[imageIndex],'score' + str(score), (10,100), font, 2,(255,255,255),2,cv.LINE_AA)
				cv.putText(paintingsScaled[imageIndex], zaal, (10,45), font, 2,(255,255,255),2,cv.LINE_AA)
				cv.imshow("image " + str(imageIndex), paintingsScaled[imageIndex])
			print zaal + "    => " + str(score)

	print ""
	imageIndex += 1;
	
cv.waitKey(0)





			# msg1 = 'using %s with lowe_ratio %.2f' % (method, lowe_ratio)
			# msg2 = 'there are %d good matches' % (len(good))
			# print msg2
			# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)

			# font = cv.FONT_HERSHEY_SIMPLEX
			# cv.putText(img3,msg1,(10, 250), font, 0.5,(255,255,255),1,cv.LINE_AA)
			# cv.putText(img3,msg2,(10, 270), font, 0.5,(255,255,255),1,cv.LINE_AA)
			# #fname = 'output_%s_%.2f.png' % (method, magic_number)
			# #cv.imwrite(fname, img3)

			# plt.imshow(img3),plt.show()