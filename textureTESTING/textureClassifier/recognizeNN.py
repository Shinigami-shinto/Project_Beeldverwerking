# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier


import numpy


from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from pyimagesearch.colorFeatures import ColorFeatures
from sklearn.svm import LinearSVC


from imutils import paths
import argparse
import cv2
import os

# fix random seed for reproducibility
numpy.random.seed(7)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=True,
	help="path to the tesitng images")
args = vars(ap.parse_args())

lbpPointCount = 10

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(lbpPointCount, 3)
colorFeatures = ColorFeatures()

data = numpy.empty((0,lbpPointCount+5), float)
labels = []

# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	hist = colorFeatures.addFeatures(image, hist)

	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split(os.path.sep)[-2])
	
	data = numpy.append(data, [hist], axis=0)



encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print labels
print encoded_Y
print dummy_y

def baseline_model():
	model = Sequential()
	model.add(Dense(40, input_dim=lbpPointCount+5, activation='relu')) #5points extra o.a. voor r g b waarden
	model.add(Dense(20, activation='relu')) #5points extra o.a. voor r g b waarden
	model.add(Dense(2, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

estimator.fit(data, dummy_y)

# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	hist = colorFeatures.addFeatures(image, hist)
	prediction = estimator.predict(hist.reshape(1, -1))
	predictedLabel = encoder.inverse_transform(prediction)[0]
	# display the image and the prediction
	cv2.putText(image, predictedLabel, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
