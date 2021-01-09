import argparse
import os
import numpy as np
import cv2 as cv
import utils
from multiprocessing.pool import ThreadPool
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False,
	help="path to the binary file (database)")
args = vars(ap.parse_args())
if not args["source"]:
	print("You must specify a file as argument with -s")
	sys.exit()
    
font = cv.FONT_HERSHEY_SIMPLEX
pt = ThreadPool(6)
img = cv.imread(args["source"])
f = args["source"]
if "/" in f:
	f = f.split("/")[-1]
utils.initialize_database()
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
			predictionPerPainting = predictionPerPaintingSorted[0]
		cv.putText(img,predictionPerPainting[1],(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
		cv.imshow("Prediction",img)
		cv.waitKey(0)
		print("We predicted:", predictionPerPainting)
	else:
		print("No possible paintings found")
else:
	print("Image is too blurry")