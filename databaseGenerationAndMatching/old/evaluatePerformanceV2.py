import argparse
import os
import numpy as np
import cv2 as cv
import cutOutPaintingsDatabase as cop
import pickle
import time
from collections import Counter
import numpy as np

size = 750 #size of the images we will compare
defaultSource = 'database.bin'

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
dirname = "/home/youssef/Documenten/Projectcomputervisie/database/zaal_2/"
f = "IMG_20190323_1120582.jpg"
def test():
    global right,total,didntFindAPainting,dirname,f
    print(f)
    #possiblePaintings = cop.cut_out_paintings(f,dirname)
    # if len(possiblePaintings) == 0:
    #     didntFindAPainting +=1
    #     print("niks gevonden")
    possiblePaintings = []
    possiblePaintings.append(cv.imread(dirname + f))
    for possiblePainting in possiblePaintings:
        #cv.imshow("image",possiblePainting)
        #cv.waitKey()
        img1 = np.copy(possiblePainting)
        possiblePainting = cv.cvtColor(possiblePainting,cv.COLOR_BGR2GRAY)
        kp,des = finder.detectAndCompute(possiblePainting,None)

        highestScores = {}
        if des is not None:
            for zaal, allDescriptorsAndnames in dataBase.items():
                #print(allDescriptorsAndnames)
                zaalBestScore = 0
                fileNameBest = ""
                zaalGoodPoints = []
                for dbFileName,descriptor in allDescriptorsAndnames:
                    score = 0
                    bf = cv.BFMatcher()
                    matches = bf.match(descriptor,des)
                    good_points = []
                    if len(matches)>0:
                        for m in matches:
                            if m.distance < 1:
                                score += 1
                                good_points.append(m)
                        if score > zaalBestScore:
                            zaalBestScore = score
                            fileNameBest = dbFileName
                            zaalGoodPoints = good_points
                highestScores[zaal] = (zaalBestScore,fileNameBest,zaalGoodPoints)
            filteredHighestScores = sorted(highestScores.items(), key=lambda x: -x[1][0])
            print("/home/youssef/Documenten/Projectcomputervisie/database/" + filteredHighestScores[0][0] + "/" + filteredHighestScores[0][1][1])
            print(filteredHighestScores[0][1][0])
            img2 = cv.imread("/home/youssef/Documenten/Projectcomputervisie/database/" + filteredHighestScores[0][0] + "/" + filteredHighestScores[0][1][1])
            kp2,des2 = finder.detectAndCompute(img2,None)
            result = cv.drawMatches(img1,kp,img2,kp2,filteredHighestScores[0][1][2],None)
            cv.imshow("result",result)
            print(zaalBestScore)
            cv.waitKey(0)
    total += 1

test()

print("right: ",right)
print("wrong: ",(total - didntFindAPainting) - right)
print("didnt find anything: ", didntFindAPainting)
print("total: ",total)

                    







