# Evaluate ML for MNIST Kaggle
#
# USAGE
# python3 eval_train_classify.py --dataset data/train.csv

# SVM
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
# Decision Tree
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
# Ramdom Forest Classifier
from sklearn.ensemble import RandomForestClassifier

import time
import numpy as np
import cv2
import argparse

from utils.hog import HOG

# split the data in training and test sets according to fraction
def splitdata_train_test(data, fraction_training):

  	#ramdomize dataset order
  	np.random.seed(0)
  	np.random.shuffle(data)

  	split = int(data.shape[0]*fraction_training)

  	training_set = data[:split]
  	testing_set = data[split:]

  	return training_set, testing_set

# Accuracy
def calculate_accuracy(predicted, actual):

  count = 0
  for i in range(len(predicted)):
    if (predicted[i] == actual[i]):
      count += 1

  return (count / len(actual))

# Main
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to the dataset file")
args = vars(ap.parse_args())

# load the dataset and initialize the data matrix
data = np.genfromtxt(args["dataset"], delimiter = ",", dtype = "uint8")
target = data[1:, 0]
digits = data[1:, 1:].reshape(data.shape[0] - 1, 28, 28) # remove head line

# Split for 70% train / 30% test
(digits_train, digits_test) = splitdata_train_test(digits, 0.7)
(target_train, target_test) = splitdata_train_test(target, 0.7)

# Some sanity check
print("Train data shape", digits_train.shape, "Target shape", target_train.shape)
print("Unique elements in targets: ", (np.unique(target_train)))

# HOG parameters to explore
intervalPPC = [2, 3, 4]                  # Pixels Per Cell (7-> very bad 0.60 or error)
intervalOrient = [2, 3, 4, 6, 18, 24]     # Orientations
intervalCPB = [1, 2, 3, 4, 5]                  # Cels Per Block

# ML Algorithms
mlAlgoList = ["SVM", "DTC", "Random Forest"]

# statistics 
avgScore = {}
avgTime = {}
maxScore = {}
totalPass = 0

for algo in mlAlgoList:
	avgScore[algo] = 0
	avgTime[algo] = 0
	maxScore[algo] = 0

# Output header
print ("Feature: HOG")
print (" _______________________________________________________________________________________________")
print ("|    ML Algo    | orientations  | pixelsPerCell | cellsPerBlock |     Score     |     Time      |")

# Loop through HOG parameters and calculate features for each model and predict
for i in intervalOrient:

	orientations = i

	for j in intervalPPC:

		hogPixelsPerCell = (j, j)

		for k in intervalCPB:

			cellsPerBlock = (k, k)

			for algo in mlAlgoList:

				start = time.perf_counter()  # Measure time

				# initialize the HOG descriptor for the variations of parameters
				hog = HOG(orientations = orientations, pixelsPerCell = hogPixelsPerCell,
						  cellsPerBlock = cellsPerBlock, block_norm = 'L2-Hys')

				data = [] # Clear data
				
				# Create model
				for image in digits_train:

					# pre-process image here if needed

					# describe the image and update the data matrix
					hist = hog.describe(image)
					data.append(hist)

				# Calculate each model for this HOG descriptor configuration
				if (algo == "SVM"):				
					model = LinearSVC(random_state = 42) # train the SVM model
				elif (algo == "DTC"):
					model = DecisionTreeClassifier()     # train DTC model
				elif(algo == "Random Forest"):
					model = RandomForestClassifier(n_estimators = 50) # train RTC model

				# Fit to the selected model
				model.fit(data, target_train)

				data = [] # Clear data

				# Prepare images and calculates features (HOG)
				for image in digits_test:

					# pre-process image here if needed

					# describe the image and update the data matrix
					hist = hog.describe(image)			
					data.append(hist)

				# Create predictions
				predicted = model.predict(data)

				# Calculate score for selected predition
				modelScore = calculate_accuracy(predicted, target_test)

				endTime = time.perf_counter() - start

				# Gather statistics
				avgScore[algo] += modelScore
				avgTime[algo] += endTime
				if (maxScore[algo] < modelScore):
					maxScore[algo] = modelScore

				totalPass += 1

				print ("|{:^15s}|{:^15d}|{:^15s}|{:^15s}|{:^15.4f}|{:^15.2f}|".format(algo, orientations, str(hogPixelsPerCell), str(cellsPerBlock), modelScore, endTime))

print (" _______________________________________________________________________________________________")

# Print summary
totalPass /= len(mlAlgoList)
print("Summary totalPass each: 	{:2.0f}".format(totalPass))
				
# Average score for each one
print("_______AVG Score________")
for i in avgScore.keys():
	print("[{:^15s}]: {:^6.2f}".format(i, avgScore[i]/totalPass))

# Average time for each one
print("_______AVG Time_________")
for i in avgTime.keys():
	print("[{:^15s}]: {:^6.2f}".format(i, avgTime[i]/totalPass))

# Max score for each one
print("_______Max Score________")
for i in maxScore.keys():
	print("[{:^15s}]: {:^6.4f}".format(i, maxScore[i]))
