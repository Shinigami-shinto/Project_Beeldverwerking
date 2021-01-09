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
print("getting descriptors from dataset..")

defaultSource = '/home/youssef/Documenten/Projectcomputervisie/testImages/testSetPerZaal/'
# defaultSource = '/home/youssef/Documenten/Projectcomputervisie/testImages/minerva/perZaal'


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False,
	help="path to the source images")
args = vars(ap.parse_args())
if not args["source"]:
	args["source"]= defaultSource
srcDir = args["source"];
print(srcDir);

dataBase = {};
for x in os.walk(srcDir):#de zalen overlopen
	if x[0].split("/")[-1].startswith('.'):
		continue
	zaal = x[0].split('/')[-1]
	zaal = zaal.lower()
	#x[2] bevat een array van alle images in een zaalDirectory
	for f in sorted(x[2], key=len): #alle fotos van de zalen overlopen:
#		print(f)
		if f.startswith('.'):
			continue
		dataBase[f] = zaal
print("dataBaseTest")
print(dataBase)
print ("\nsaving binary..")

with open('testlabels.bin', 'wb') as handle:
    pickle.dump(dataBase, handle, protocol=pickle.HIGHEST_PROTOCOL)

print( "loading binary (for testing)")

with open('testlabels.bin', 'rb') as handle:
    dataBase = pickle.load(handle, encoding='latin1')

print( "done")




