import argparse
import os
import cv2
import random

defaultSource = '/Users/emielPC/Google Drive/unif/computerVisie/dataset/dataset_pictures_msk_nokia7plus'
defaultDestination = 'patches'
patchSize = 100;

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

allImages = [];
for x in os.walk(srcDir):
	#x[2] bevat een array van alle images in een zaalDirectory
	for f in x[2]:
		allImages.append(x[0]+"/"+f)

maxi = len(allImages)

mouseX = 0
mouseY = 0
srcScaled = []
scale = 1;
def mousePosition(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
    	global mouseX,mouseY,srcScaled,scale
        mouseX = x;
        mouseY = y;
        patchScaled = int(patchSize / 2.0 * scale);
    	if len(srcScaled)>0:
    		srcScaledCop = srcScaled.copy()
        	cv2.rectangle(srcScaled,(mouseX - patchScaled, mouseY - patchScaled),(mouseX+patchScaled,mouseY+patchScaled),(0,255,0),1)
        	cv2.imshow("image", srcScaled)
        	srcScaled = srcScaledCop;

cv2.namedWindow('image')
cv2.setMouseCallback('image',mousePosition)


key = ord('a') #key een random value geven
#'s' to skip an image
#'q' to quit
#other letters to make classes
while key != ord('q'):
	src = cv2.imread(cv2.samples.findFile(allImages[int(random.random() * (maxi - 1))]))
	height, width, channels = src.shape;
	scale = min(1260.0/width, 720.0 / height)
	srcScaled = cv2.resize(src, (int(width*scale), int(height*scale)))
	cv2.imshow("image", srcScaled)
	
	key = cv2.waitKey(0)
	while key != ord('q') and key != ord('s'):
		locX, locY = int(mouseX/scale) , int(mouseY/scale)

		#buiten de randen klick opvangen
		verschuivingY = 0
		verschuivingX = 0
		if locY - patchSize/2 < 0:
			verschuivingY = -(locY - patchSize/2)
		if locY + patchSize/2 > height:
			verschuivingY = height - (locY + patchSize/2)
		if locX - patchSize/2 < 0:
			verschuivingX = -(locX - patchSize/2)
		if locX + patchSize/2 > width:
			verschuivingX = width - (locX + patchSize/2)

		patch = src[locY + verschuivingY - patchSize/2:locY + verschuivingY + patchSize/2,
		 locX + verschuivingX - patchSize/2:locX + verschuivingX + patchSize/2]
		cv2.imshow("imagPatch", patch)

		folder = destDir + "/" + chr(key)

		if not os.path.exists(folder):
			os.makedirs(folder)

		newFileName = folder + "/" + chr(key) + "-" + str(int(random.random() * 1000000)) + ".jpg"
		#newFileName = chr(key) + "-" + str(int(random.random() * 1000000)) + ".jpg"
		print(newFileName)
		cv2.imwrite(newFileName, patch, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
		key = cv2.waitKey(0)














