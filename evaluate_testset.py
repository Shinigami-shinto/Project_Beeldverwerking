import argparse
import os
import numpy as np
import cv2 as cv
import utils
import pickle
import time
from multiprocessing.pool import ThreadPool

#metrics
TP=FP=TN=FN=cutOutDidntWork=tooBlurryPictures=0
dirname =""

def handle_file(f):
	global cutOutDidntWork,tooBlurryPictures, dirname
	pt = ThreadPool(2)
	if f.startswith('.'):
		return []
	dirname += "/"
	inputImage = cv.imread(dirname+f)
	if utils.is_too_blurry(inputImage):
		tooBlurryPictures+=1
		print(f,"too blurry")
		return []
	paintings = utils.cut_out_paintings(inputImage)
	inputDescriptors = pt.map(utils.get_desc_and_keypoints,paintings)

	if len(paintings)==0: cutOutDidntWork+=1

	scores = pt.map(utils.match_with_db, inputDescriptors)
	predictionPerPainting = []
	for topScoresImage in scores:
		if len(topScoresImage) != 0:
			topscoresSorted = sorted(topScoresImage.items(), key=lambda x: -x[1])
			score = topscoresSorted[0][1]
			besteZaal = topscoresSorted[0][0]
			if score == 0:
				besteZaal = "geen painting"
			predictionPerPainting.append((f,besteZaal,score))
		else:
			predictionPerPainting.append((f,"geen painting",0))
	if(len(predictionPerPainting) != 0):
		predictionPerPaintingSorted = sorted(predictionPerPainting, key=lambda x: -x[2])
		predictionPerPainting = [predictionPerPaintingSorted[0]]
	return predictionPerPainting

defaultPath = '/home/youssef/Documenten/Projectcomputervisie/testImages/testSetPerZaal/'
path_single_paintings_abstract = "D:/School/2018-2019/Project CV - Paintings/RAW dataset CV_AJ1819/pictures_msk_smak_galaxy_A5/single_paintings/houses_villages/"

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False,
	help="path to the binary file (database)")
args = vars(ap.parse_args())
if not args["source"]:
	args["source"]= defaultPath
srcPath = args["source"]

#opencl aanzetten:
print("opencl?")
print("uses opencl first: " + str(cv.ocl.useOpenCL()))
print("has opencl: " + str(cv.ocl.haveOpenCL()))
cv.ocl.setUseOpenCL(True)
print("uses opencl: " + str(cv.ocl.useOpenCL()))
    
font = cv.FONT_HERSHEY_SIMPLEX

def TESTING():#gaat alle testafbeeldingen af en geeft telkens alle votes
	global TP,FP,TN,FN,cutOutDidntWork, dirname
	p = ThreadPool(3)

	for dr, dirnames, filenames in os.walk(srcPath):
		print(dr)
		dirname = dr
		predictions = p.map(handle_file,filenames)
		for arr in predictions:
			for fileName,pred,score in arr:
				print("we predicted:",pred,"with a score:",score,"for img",fileName,". The ground truth is:",dataBaseTest[fileName])
				if score != 0:
					if(dataBaseTest[fileName] == pred):
						TP += 1
					elif(dataBaseTest[fileName] != pred):
						FP += 1
				else:
					if(dataBaseTest[fileName] == pred):
						TN += 1
					elif(dataBaseTest[fileName] != pred):
						FN += 1
utils.initialize_database()
dataBaseTest = utils.initialize_testlabels()
TESTING()

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP+TN+FP+FN)
print("\n\nPrecision:            " + str(precision))
print("\nRecall:            " + str(recall))
print("\nAccuracy:            " + str(accuracy))
print("\nNumber of images where cutOut didn't find anything: ",cutOutDidntWork)
print("\nNumber of images where too blurry: ",tooBlurryPictures)