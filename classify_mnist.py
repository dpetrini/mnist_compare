# USAGE
# python classify_mnist.py --model models/svm.cpickle --test data/testexxx.csv

# import the necessary packages
from __future__ import print_function
from sklearn.externals import joblib
from utils.hog import HOG
import sys
import argparse
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model will be stored")
ap.add_argument("-t", "--test", required = True,
	help = "path to the image file")
args = vars(ap.parse_args())

sys.stdout=open("mnist_classify_output-003.txt","w")

# load the dataset and initialize the data matrix
data = np.genfromtxt(args["test"], delimiter = ",", dtype = "uint8")
print(data.shape)
digits_test = data[1:, :].reshape(data.shape[0] - 1, 28, 28) # remove head line

# Some sanity check
print("Test data shape", digits_test.shape)

# load the model
model = joblib.load(args["model"])

# initialize the HOG descriptor
hog = HOG(orientations = 3, pixelsPerCell = (2, 2),
	cellsPerBlock = (4, 4), block_norm = 'L2-Hys')

data = []

# Prepare images and calculates features (HOG)
for image in digits_test:

	# describe the image and update the data matrix
	hist = hog.describe(image)			
	data.append(hist)

# Create predictions
predicted = model.predict(data)

print("Predict shape", predicted.shape, " Size: ", len(predicted))
print("ImageId,Label")

counter = 1;
for i in predicted:
	print("{},{}".format(counter,i))
	counter += 1


sys.stdout.close()
