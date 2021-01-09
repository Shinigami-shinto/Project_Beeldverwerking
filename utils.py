import cv2
import numpy as np
import pickle
import math
import random

#Threshold
BLURTHRES = 15
maxHammingDistance = 32

dataBase = {}

def initialize_database():
    #db laden:
    global dataBase
    with open('database.bin', 'rb') as handle:
        dataBase = pickle.load(handle, encoding='latin1')

def initialize_testlabels():
    with open('testlabels.bin', 'rb') as handle:
        testlabels = pickle.load(handle, encoding='latin1')
    return testlabels

#get a square from countour points:
def get_rect_from_contourPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    x = pts[:,0]
    y = pts[:,1]
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    rect[1] = (xmax,ymin)
    rect[3] = (xmin,ymax)
    rect[0] = (xmin,ymin)
    rect[2] = (xmax,ymax)

    # return the ordered coordinates
    return rect

#cut the rect out of the image:
def cut_rect_out_of_image(image, pts):
    rect = get_rect_from_contourPoints(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

#Kijkt hoeveel % van de punten aan de rand liggen
def getContourBorderPercentage(contour, maxX, maxY):
    amount = 0
    borderAmount = 0.0

    for point in contour:
        if point[0] == 0 or point[1] == 0:
            borderAmount += 1
        if point[0] >= maxX -1  or point[1] >= maxY - 1:
            borderAmount += 1
        if point[1] >= maxY -1:
            borderAmount += 0.5 #onderste rand extra afstraffen, segmenten die onderaan raken hebben we zelden nodig
        amount += 1


    #print("percentage: " + str(borderAmount / float(amount)))
    return borderAmount / float(amount)

def cut_out_paintings(src):
    resultImages = []
    rows,cols,_ = src.shape
    if rows*cols > 1920*1080:
        src = cv2.resize(src, (cols//3, rows//3))
    src_blurred = cv2.GaussianBlur(src,(21,21),0)
    
    #Perform mean shift segmentation on the image
    spatial_radius = 7
    color_radius = 13
    
    meanS = cv2.pyrMeanShiftFiltering(src_blurred, spatial_radius, color_radius, maxLevel=1)

    src_copy = np.copy(meanS)
    rows,cols,ch = src_copy.shape
    mask = np.zeros((rows+2,cols+2),dtype=np.uint8)
    loDiff = (1,1,1)
    hiDiff = (1,1,1)

    #Find largest segment with same color
    wallColor = [0,0,0]
    largestSegment = 0
    for y in range(rows):
            for x in range(cols):
                    if mask[y+1][x+1] == 0:
                            newVal = (int(random.random() * 255),int(random.random() * 255),int(random.random() * 255))
                            size, _, _, _ = cv2.floodFill(src_copy,mask,(x,y),newVal,loDiff,hiDiff,4)
                            if size > largestSegment:
                                    largestSegment = size
                                    wallColor = newVal

    #Make mask with this largest segment
    mask = cv2.inRange(src_copy,wallColor,wallColor)

    #Dialate
    mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=6)
    #Invert
    mask = cv2.bitwise_not(mask)
    #########als code errort op de lijn hieronder, verander door de andere:
    _ ,contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 12000:
            a = contour.ravel()
            a= a.reshape(len(contour),2)

            #0.33 = 33% van de countour lig op de rand van de afbeelding
            borderPercentage = getContourBorderPercentage(a, cols, rows)
            if borderPercentage < 0.4: #max 40% op de rand!
                rect = get_rect_from_contourPoints(a)
                img = cut_rect_out_of_image(src,rect)
                resultImages.append(img)
    return resultImages

    
def real_time(img):
    #src = cv2.resize(img, (1310, 720))
    src = img
    src_blurred = cv2.GaussianBlur(src,(5,5),0)

    src_copy = np.copy(src_blurred)
    rows,cols,ch = src_copy.shape

    mask = np.zeros((rows+2,cols+2),dtype=np.uint8)
    loDiff = (1,1,1)
    hiDiff = (1,1,1)

    wallColor = [0,0,0]
    largestSegment = 0
    for y in range(rows):
            for x in range(cols):
                    if mask[y+1][x+1] == 0:
                            newVal = (int(random.random() * 255),int(random.random() * 255),int(random.random() * 255))
                            size, _, _, _ = cv2.floodFill(src_copy,mask,(x,y),newVal,loDiff,hiDiff,4)
                            if size > largestSegment:
                                    largestSegment = size
                                    wallColor = newVal

    mask = cv2.inRange(src_copy,wallColor,wallColor)
    #cv2.imshow("src_copy", src_copy)

    # mask = cv2.inRange(hsv,lower_background,higher_background)
    # mask = cv2.inRange(src,lower_background,higher_background)
    #Dialate
    mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=6)
    #Invert
    mask = cv2.bitwise_not(mask)

    #########als code errort op de lijn hieronder, verander door de andere:
    #########_ ,contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    _,contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    i =0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 6000:
            a = contour.ravel()
            #reshape (x,y)(x,y)...
            a= a.reshape(len(contour),2)

            rect = get_rect_from_contourPoints(a)
            img = cut_rect_out_of_image(src,rect)

            cv2.imshow("src" + str(i), img)
            i +=1
            cv2.drawContours(src,contour,-1,(0,255,0),3)


    #imS = cv2.resize(mask, (960, 540))
    #src = cv2.resize(src, (960, 540))
    #cv2.imshow("hsv", imS)
    #cv2.imshow("src", src)
    return src

def match_with_db(inputDesc):
	global maxHammingDistance
	topScore = {}
	if(inputDesc is not None):
		for zaal, allDescriptors in dataBase.items(): #loop over alle zalen in de db
			zaalBestScore = 0
			for dbFileName,descriptor in allDescriptors: #loop over alle schilderijen in die zaal
				# BFMatcher with default params
				matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
				matches = matcher.match(descriptor,inputDesc)
				goodMatches = filter(lambda m: m.distance <= maxHammingDistance, matches)
				goodMatches = sorted(goodMatches, key=lambda x: x.distance)

				if len(goodMatches) > zaalBestScore:
					zaalBestScore = len(goodMatches)
			#de score van de zaal wordt de score van de beste afbeelding in die map
			topScore[zaal] = zaalBestScore
	return topScore

def variance_of_laplacian(img):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(img, cv2.CV_64F).var()

def is_too_blurry(img):
    variance = variance_of_laplacian(img)
    if variance < BLURTHRES:
        return True
    return False

finder = None
def get_desc_and_keypoints(painting):
    global finder
    #initializing keypoint creator
    if finder is None:
        finder = cv2.ORB_create()
    #img = cv.resize(img, (size, size))
    img = cv2.cvtColor(painting,cv2.COLOR_BGR2GRAY)
    #cv.imshow("cut",img)
    #cv.waitKey()
    kp2, des2 = finder.detectAndCompute(img,None)
    return des2