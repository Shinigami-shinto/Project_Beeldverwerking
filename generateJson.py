import argparse
import os
import pickle
import numpy as np
import cv2 as cv

#initializing keypoint creator
print ("initializing keypoint creator..")
method = 'ORB'  # 'SIFT'
if method   == 'ORB':
    finder = cv.ORB_create()
elif method == 'SIFT':
    finder = cv.xfeatures2d.SIFT_create()

#alle descriptoren van alle afbeeldingen ophalen en groeperen per zaal
print ("getting descriptors from dataset..")

defaultSource = '/home/youssef/Documenten/Projectcomputervisie/database/'
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False,
	help="path to the source images")
args = vars(ap.parse_args())
if not args["source"]:
	args["source"]= defaultSource
srcDir = args["source"];
print(srcDir);

dataBase = {};
des1 = "hallo";
ignoreFirst = True
for x in os.walk(srcDir):#de zalen overlopen
	if ignoreFirst:#ignore the top level map
		ignoreFirst = False
		continue
	if x[0].split("/")[-1].startswith('.'):
		continue
	zaal = x[0].split('/')[-1]
	zaal = zaal.lower()
	dataBase[zaal] = []
	#x[2] bevat een array van alle images in een zaalDirectory
	for f in x[2]: #alle fotos van de zalen overlopen:
		if f.startswith('.'):
			continue
		src = cv.imread(x[0] + "/" + f,0) # queryImage
		kp1, des1 = finder.detectAndCompute(src,None)
		if des1 is not None:
			dataBase[zaal].append((f,des1));


print ("saving binary..")

with open('database.bin', 'wb') as handle:
    pickle.dump(dataBase, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ("loading binary (for testing)")

with open('database.bin', 'rb') as handle:
    dataBase = pickle.load(handle)

print ("done")




