# Compare descriptors and Machine Learning Algorithms
Compare different HOG descriptor parameters and ML approaches for image classification.

The python script __eval_train_classify.py__ perform following tasks:
 * read all dataset, split in train and test 
 * create HOG descritors for all train dataset for a variation of HOG parameters
 * create models for different machine learning algorithm like SVM, random forest
 * classify immediately the split testset and get accuracy of model
 * create a table with all results

The objective is to use the table to decide the best algorithm and HOG parameters to use for solve your problem.
Here the image dataset is MNIST, the hand-writing Digits Dataset. You can use any image dataset with few modifications.

## MNIST Dataset
MNIST is a widely used dataset for computer vision. It has tenths of thousands of labeled digits in gray images of 28x28 pixels size.
Moreover there is a test set, it means a also large set of unlabeled images (in Kaggle) so that one can test its model.
The dataset used in this repository is from https://www.kaggle.com/c/digit-recognizer/data.
(download the files train.csv and test.csv and place in data/ directory)

Example of MNIST digit:

<img src="example/digit.jpg" width="200">

## Image Classification 
One approach to perform MNIST classification is through HOG descriptors.

HOG means Histogram of Oriented Gradients. It generates a histogram of gradients of images countours and edges.
With this image descriptor we can compare images and check whether they are similar (same digit)
or not.

Machine learning algorithm takes many examples (train dataset) of digit images, digit label (the 
digit name) and the HOG descriptor of many samples and learn from it.

After learning model is created, one can predict new digits label (values) from the test dataset.

Then we compare the real values with the predicted values. With these values we calculate accuracy.
The table below shows the accuracy of each Machine Learning algorithm in this list:

 * SVM - Support Vector Machine
 * DTC - Decision Tree Classifier
 * Random Forest - a set of decision trees split for better prediction
 * (to increase soon)

And many different configurations of HOG descriptor with following parameters:

 * Orientations
 * Pixels Per Cell
 * Cells Per Block

As we can see, changing this parameters affects completely in accuracy.

With this tool one can compare image based classifications based in HOG and find the best result to 
train an actual model for its system and then predict.

## Technical Stack
* Python
* sklearn

## Usage
Main script: evaluate by training and classifying at same time. Generates a report. It may take a while...
```
python3 eval_train_classify.py --dataset data/train.csv
```

After you selected your parameter and algorithm based in results of evaluation, you can train your dataset:
```
python3 train.py --dataset data/train.csv --model models/rf.cpickle
```

And classify the test vector:
```
python3 classify_mnist.py --model models/rf.cpickle --test data/test.csv
```

## Output from eval_train_classify.py (performance table)

Train data shape (29399, 28, 28) Target shape (29399,)
Unique elements in targets:  [0 1 2 3 4 5 6 7 8 9]
Feature: HOG


|    ML Algo    | orientations  | pixelsPerCell | cellsPerBlock |     Score     |     Time      |
|---------------|---------------|---------------|---------------|---------------|---------------|
|      SVM      |       2       |    (2, 2)     |    (1, 1)     |    0.5677     |    447.74     |
|      DTC      |       2       |    (2, 2)     |    (1, 1)     |    0.8107     |    402.42     |
| Random Forest |       2       |    (2, 2)     |    (1, 1)     |    0.9475     |    413.58     |
|      SVM      |       2       |    (2, 2)     |    (2, 2)     |    0.7965     |    496.72     |
|      DTC      |       2       |    (2, 2)     |    (2, 2)     |    0.8193     |    480.40     |
| Random Forest |       2       |    (2, 2)     |    (2, 2)     |    0.9521     |    417.86     |
|      SVM      |       2       |    (2, 2)     |    (3, 3)     |    0.8219     |    404.21     |
|      DTC      |       2       |    (2, 2)     |    (3, 3)     |    0.8193     |    416.96     |
| Random Forest |       2       |    (2, 2)     |    (3, 3)     |    0.9556     |    376.84     |
|      SVM      |       2       |    (2, 2)     |    (4, 4)     |    0.8322     |    407.08     |
|      DTC      |       2       |    (2, 2)     |    (4, 4)     |    0.8180     |    416.41     |
| Random Forest |       2       |    (2, 2)     |    (4, 4)     |    0.9520     |    340.62     |
... (lines ommited for brevity, etc, etc)
| Random Forest |       6       |    (4, 4)     |    (3, 3)     |    0.9683     |     95.93     |
|      SVM      |       6       |    (4, 4)     |    (4, 4)     |    0.8986     |     63.41     |
|      DTC      |       6       |    (4, 4)     |    (4, 4)     |    0.8573     |     99.39     |
... best result:
|__Random Forest__|  __6__      |  __(4, 4)__   |  __(4, 4)__   |  __0.9701__   |   __79.80__   |
|      SVM      |       6       |    (4, 4)     |    (5, 5)     |    0.8772     |     47.23     |
|      DTC      |       6       |    (4, 4)     |    (5, 5)     |    0.8548     |     72.41     |
| Random Forest |       6       |    (4, 4)     |    (5, 5)     |    0.9695     |     60.55     |
|      SVM      |      18       |    (2, 2)     |    (1, 1)     |    0.8022     |    415.57     |
...
| Random Forest |      24       |    (4, 4)     |    (1, 1)     |    0.9541     |    118.87     |
|      SVM      |      24       |    (4, 4)     |    (2, 2)     |    0.9225     |    308.32     |
|      DTC      |      24       |    (4, 4)     |    (2, 2)     |    0.8289     |    158.07     |
| Random Forest |      24       |    (4, 4)     |    (2, 2)     |    0.9607     |    124.95     |
|      SVM      |      24       |    (4, 4)     |    (3, 3)     |    0.9333     |    227.19     |
|      DTC      |      24       |    (4, 4)     |    (3, 3)     |    0.8346     |    196.75     |
| Random Forest |      24       |    (4, 4)     |    (3, 3)     |    0.9588     |    117.16     |
|      SVM      |      24       |    (4, 4)     |    (4, 4)     |    0.9114     |     81.49     |
|      DTC      |      24       |    (4, 4)     |    (4, 4)     |    0.8286     |    200.64     |
| Random Forest |      24       |    (4, 4)     |    (4, 4)     |    0.9600     |     99.57     |
|      SVM      |      24       |    (4, 4)     |    (5, 5)     |    0.8918     |     63.80     |
|      DTC      |      24       |    (4, 4)     |    (5, 5)     |    0.8283     |    163.17     |
| Random Forest |      24       |    (4, 4)     |    (5, 5)     |    0.9618     |     80.01     |
 _______________________________________________________________________________________________
Summary totalPass each: 	90 (many lines ommited in above table for brevity, see file: eval_output.txt for complete result)

|__AVG Score__  |       |
|---------------|-------|
|      SVM      |  0.85 |
|      DTC      |  0.82 |
| Random Forest |  0.95 |

|__AVG Time__   |       |
|---------------|-------|
|      SVM      | 281.85|
|      DTC      | 332.58|
| Random Forest | 219.89|

|__Max Score__  |       |
|---------------|-------|
|      SVM      | 0.9333|
|      DTC      | 0.8752|
| Random Forest | 0.9707|

## Conclusion

The time measurement is just a reference, will be different in other machines.

Use this script to find the best algorithm to use and the best sklearn HOG descriptor parameters. Then create model assured that is the best performance combination.
