import argparse
import os
import cv2
import random

size = (750,750) #size of the output images
defaultSource = 'cuttedImages'
defaultDestination = 'scaledCuttedImages'



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--destination", required=False,
	help="path where the patches can be saved")
ap.add_argument("-s", "--source", required=False,
	help="path to the source images")
args = vars(ap.parse_args())

if not args["source"]:
	args["source"]= defaultSource
if not args["destination"]:
	args["destination"] = defaultDestination
srcDir = args["source"];
destDir = args["destination"]
print(srcDir);
print(destDir);

#desitnation map aanmaken indien nodig:


for x in os.walk(srcDir): #alle zalen overlopen:
	if x[0].split("/")[-1].startswith('.'):
		continue
	zaal = x[0].split('/')[-1]
	#x[2] bevat een array van alle images in een zaalDirectory
	for f in x[2]: #alle bestanden in deze zalen overlopen
		if f.startswith('.'):#begint met een punt is een verborgen file.. skippen
			continue

		saveToFolder = destDir + "/" + zaal + "/"
		if not os.path.exists(saveToFolder):
			os.makedirs(saveToFolder)

		src = cv2.imread(cv2.samples.findFile(x[0] + "/" + f))
		src = cv2.resize(src, size)
		cv2.imwrite(saveToFolder + f, src, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
	









