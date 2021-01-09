import numpy as np

class ColorFeatures:

		

	def addFeatures(self, image, hist):
		src = image.copy();
		height, width, channels = src.shape
		totalPix = height * width;
		src = src.astype("float")
		src /= 255.0;
		r = 0
		g = 0
		b = 0
		for y in range(0,height):
			for x in range(0,width):
				pix = src[y,x]
				r += pix[0]
				g += pix[1]
				b += pix[2]
		r /= float(totalPix)
		g /= float(totalPix)
		b /= float(totalPix)
		#print "avg rgb:"
		#print r,g,b
	
		hist = np.append(hist, [r,g,b])

		#todo: standard deviations of the colors ook toevoegen aan de features

		# return the histogram of Local Binary Patterns
		return hist