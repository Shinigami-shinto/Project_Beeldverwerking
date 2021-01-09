import argparse
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import cutOutPaintings as cop
import pickle
import time

size = 750 #size of the images we will compare
defaultSource = 'database.bin'

path_selfgen_test = "D:/School/2018-2019/Project CV - Paintings/Testimgs_copy/"
path_single_paintings_abstract = "D:/School/2018-2019/Project CV - Paintings/RAW dataset CV_AJ1819/pictures_msk_smak_galaxy_A5/single_paintings/abstract/"


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False,
	help="path to the binary file (database)")
args = vars(ap.parse_args())
if not args["source"]:
	args["source"]= defaultSource
srcFile = args["source"];
print(srcFile);


dataBase = {};
with open('database.bin', 'rb') as handle:
    dataBase = pickle.load(handle, encoding='latin1')


#initializing keypoint creator
print("initializing keypoint creator..")
method = 'ORB'  # 'SIFT'
lowe_ratio = 0.89
if method   == 'ORB':
    finder = cv.ORB_create()
elif method == 'SIFT':
    finder = cv.xfeatures2d.SIFT_create()
    
def TESTING():#gaat alle testafbeeldingen af en geeft telkens alle votes
    for files in os.listdir(path_single_paintings_abstract):
        print(files)
        start = time.time()
        paintings = cop.cut_out_paintings(files , path_single_paintings_abstract)
        inputDescriptors = []
        for img in paintings:
        	img = cv.resize(img, (size, size))
        	kp2, des2 = finder.detectAndCompute(img,None)
        
        	inputDescriptors.append(des2)
        
        
        topScores = {}
        
        for zaal, allDescriptors in dataBase.items(): #loop over alle zalen in de db
        # find the keypoints and descriptors with SIFT
            zaalBestScore = 0
            for descriptors in allDescriptors: #loop over alle schilderijen in die zaal
                for inputDesc in inputDescriptors: #loop over alle schilderijen in de inputfoto
                    if(inputDesc is not None):
                        score = 0
                        
                        # BFMatcher with default params
                        bf = cv.BFMatcher()
                        #type = ndarrays
#                        print("\ntroubleshooting types, rows and cols: ")
#                        print("\nshape desc: ")
#                        print(descriptors.shape)
#                        print("\nshape inputdesc: ")
#                        print(inputDesc.shape)
                        
                        
                        matches = bf.knnMatch(descriptors,inputDesc, k=2)
            
                        # Apply ratio test
                        good = []
            
                        for m,n in matches:
                            if m.distance < lowe_ratio*n.distance:
                                good.append([m])
                                score += 10.0 / m.distance
                        if score > zaalBestScore:
                            zaalBestScore = score
            #de score van de zaal wordt de score van de beste afbeelding in die map
            topScores[zaal] = zaalBestScore
        
        done = time.time()
        elapsed = done-start
        print("\nTime to cut_out and compare to DB of keypoint and descriptors: ")
        print(round(elapsed,2))
#        if zaalBestScore > 100:
        for zaal, score in sorted(topScores.items(), key=lambda x: x[1]):
            print(zaal + "    => " + str(score))
#        else:print("geen match, score te laag!!")

print("\nTesting starting now!")
TESTING()
