import argparse
import os
import numpy as np
import cv2 as cv
import cutOutPaintingsDatabase as cop
import pickle
import time
from collections import Counter

size = 750 #size of the images we will compare
defaultSource = 'database.bin'

path_selfgen_test = "/home/youssef/Documenten/Projectcomputervisie/testImages/minerva/perZaal"
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
right=total=didntFindAPainting=0

def test():
    global right,total,didntFindAPainting
    for dirname, dirnames, filenames in os.walk(path_selfgen_test):
        print(len(filenames))
        for f in filenames:
            print(f)
            if f.startswith('.'):
                continue
            dirname += "/"
            possiblePaintings = cop.cut_out_paintings(f,dirname)
            if len(possiblePaintings) == 0:
                didntFindAPainting +=1
                print("niks gevonden")
            madeDecisionsPerPossiblePainting = []
            for possiblePainting in possiblePaintings:
                #cv.imshow("image",possiblePainting)
                #cv.waitKey()
                possiblePainting = cv.cvtColor(possiblePainting,cv.COLOR_BGR2GRAY)
                kp,des = finder.detectAndCompute(possiblePainting,None)

                highestScores = {}
                if des is not None:
                    for zaal, allDescriptors in dataBase.items():
                        zaalBestScore = 0
                        for dbFileName,descriptor in allDescriptors:
                            score = 0
                            bf = cv.FlannBasedMatcher(dict(algorithm=0,trees=5),dict())
                            matches = bf.knnMatch(descriptor,des, k=2)

                            if len(matches[0])>1:
                                for m,n in matches:
                                    if m.distance < lowe_ratio*n.distance:
                                        score +=10 / m.distance
                                if score > zaalBestScore:
                                    zaalBestScore = score
                        highestScores[zaal] = zaalBestScore
                    filteredHighestScores = dict(sorted(highestScores.items(), key=lambda x: -x[1]))
                    zalen = list(filteredHighestScores.keys())[:3]
                    madeDecisionsPerPossiblePainting.append(zalen)
            if len(possiblePaintings) != 0 and len(madeDecisionsPerPossiblePainting) != 0:
                madeDecisionsPerPossiblePainting = [item for sublist in madeDecisionsPerPossiblePainting for item in sublist]
                freq = Counter(madeDecisionsPerPossiblePainting)
                bestZaal = freq.most_common(1)[0][0]
                if bestZaal == dataBaseTest[f]:
                    print("is juist")
                    right +=1
                else:
                    print("oei :(")
            total += 1

test()

print("right: ",right)
print("wrong: ",(total - didntFindAPainting) - right)
print("didnt find anything: ", didntFindAPainting)
print("total: ",total)

                    







