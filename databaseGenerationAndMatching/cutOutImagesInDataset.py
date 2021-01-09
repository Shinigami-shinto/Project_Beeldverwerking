import argparse
import os
import cv2
import random
import cutOutPaintingsDatabase as cop


defaultSource = '/Users/emielPC/Google Drive/unif/computerVisie/dataset/dataset_pictures_msk_nokia7plus'
defaultDestination = 'cuttedImages'
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

for x in os.walk(srcDir):#alle zalen overlopen:
	if x[0].split("/")[-1].startswith('.'):
		continue
	zaal = x[0].split('/')[-1]
	#x[2] bevat een array van alle images in een zaalDirectory
	for f in x[2]: #alle fotos in deze zaal overlopen
		if f.startswith('.'): #bestand begint met een punt.. (skippen)
			continue
		paintings = cop.cut_out_paintings("/"+f, x[0])
		saveToFolder = destDir + "/" + zaal + "/"
		if not os.path.exists(saveToFolder):
				os.makedirs(saveToFolder)

		for p in range(0, len(paintings)): #alle paintings in dit bestand overlopen
			newFileName = f.split(".")
			newFileName[-2] += str(p); #nummer toevoegen aan de filename
			newFileName = ".".join(newFileName)
		
			cv2.imwrite(saveToFolder + newFileName, paintings[p], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
		











