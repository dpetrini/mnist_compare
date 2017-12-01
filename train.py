# USAGE
# python3 train.py --dataset data/train.csv --model models/rf.cpickle

# import the necessary packages
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from utils.hog import HOG
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to the dataset file")
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model will be stored")
args = vars(ap.parse_args())

# load the dataset and initialize the data matrix
data = np.genfromtxt(args["dataset"], delimiter = ",", dtype = "uint8")
target = data[1:, 0]
digits = data[1:, 1:].reshape(data.shape[0] - 1, 28, 28) # remove head line

data = []

# initialize the HOG descriptor with the best score from evaluation
# for 0.9707
hog = HOG(orientations = 3, pixelsPerCell = (2, 2),
	cellsPerBlock = (4, 4), block_norm = 'L2-Hys')

# loop over the images
for image in digits:

	# pre-process image here if needed

	# describe the image and update the data matrix
	hist = hog.describe(image)
	data.append(hist)

# train the model
#model = LinearSVC(random_state = 42)
model = RandomForestClassifier(n_estimators = 50) # train RTC model
model.fit(data, target)

# dump the model to file
joblib.dump(model, args["model"])
