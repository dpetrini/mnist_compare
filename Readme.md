# Compare descriptors and Machine Learning Algorithms
Compare different HOG descriptor parameters and ML approaches.

The python script eval_train_classify.py perform following tasks:
 * read all dataset, split in train and test 
 * create HOG descritors for all train dataset for a variation of HOG parameters
 * create models for different machine learning algorithm like SVM, random forest
 * classify immediately the split testset and get accuracy of model
 * create a table with all results

The objective is to use the table to decide the best algorithm and HOG parameters to use for solve your problem.
Here the image dataser is MNIST, the hand-writing Digits Dataset. You can use any image dataset with few modifications.

## MNIST Dataset
MNIST is a widely used dataset for computer vision. It has tenths of thousands of labeled digits in gray images of 28x28 pixels size.
Moreover there is a test set, it means a also large set of unlabeled images (in Kaggle) so that one can test its model.
The dataset used in this repository is from https://www.kaggle.com/c/digit-recognizer/data.

example of MNIST digit:

<img src="example/digit.png" width="200"

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
```
python3 eval_train_classify.py --dataset data/train.csv
```

After you selected your parameters, algorithm based in results of evaluation, you can train your dataset:
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
|      SVM      |       2       |    (2, 2)     |    (5, 5)     |    0.8506     |    363.25     |
|      DTC      |       2       |    (2, 2)     |    (5, 5)     |    0.8233     |    420.84     |
| Random Forest |       2       |    (2, 2)     |    (5, 5)     |    0.9548     |    308.51     |
|      SVM      |       2       |    (3, 3)     |    (1, 1)     |    0.4566     |    233.81     |
|      DTC      |       2       |    (3, 3)     |    (1, 1)     |    0.7972     |    170.29     |
| Random Forest |       2       |    (3, 3)     |    (1, 1)     |    0.9359     |    176.00     |
|      SVM      |       2       |    (3, 3)     |    (2, 2)     |    0.7987     |    326.14     |
|      DTC      |       2       |    (3, 3)     |    (2, 2)     |    0.8064     |    195.30     |
| Random Forest |       2       |    (3, 3)     |    (2, 2)     |    0.9427     |    197.80     |
|      SVM      |       2       |    (3, 3)     |    (3, 3)     |    0.8487     |    326.58     |
|      DTC      |       2       |    (3, 3)     |    (3, 3)     |    0.8060     |    151.50     |
| Random Forest |       2       |    (3, 3)     |    (3, 3)     |    0.9433     |    143.82     |
|      SVM      |       2       |    (3, 3)     |    (4, 4)     |    0.8610     |    176.11     |
|      DTC      |       2       |    (3, 3)     |    (4, 4)     |    0.8082     |    131.57     |
| Random Forest |       2       |    (3, 3)     |    (4, 4)     |    0.9447     |    119.18     |
|      SVM      |       2       |    (3, 3)     |    (5, 5)     |    0.8364     |     81.40     |
|      DTC      |       2       |    (3, 3)     |    (5, 5)     |    0.8042     |    110.37     |
| Random Forest |       2       |    (3, 3)     |    (5, 5)     |    0.9461     |     97.50     |
|      SVM      |       2       |    (4, 4)     |    (1, 1)     |    0.5052     |    172.23     |
...
| Random Forest |       2       |    (4, 4)     |    (5, 5)     |    0.9508     |     52.44     |
|      SVM      |       3       |    (2, 2)     |    (1, 1)     |    0.6707     |    401.62     |
|      DTC      |       3       |    (2, 2)     |    (1, 1)     |    0.8382     |    389.46     |
| Random Forest |       3       |    (2, 2)     |    (1, 1)     |    0.9618     |    391.26     |
...
|      SVM      |       3       |    (4, 4)     |    (3, 3)     |    0.8957     |    173.80     |
|      DTC      |       3       |    (4, 4)     |    (3, 3)     |    0.8732     |     91.56     |
| Random Forest |       3       |    (4, 4)     |    (3, 3)     |    0.9668     |     89.63     |
|      SVM      |       3       |    (4, 4)     |    (4, 4)     |    0.8584     |     59.12     |
|      DTC      |       3       |    (4, 4)     |    (4, 4)     |    0.8741     |     72.78     |
| Random Forest |       3       |    (4, 4)     |    (4, 4)     |    0.9652     |     71.22     |
|      SVM      |       3       |    (4, 4)     |    (5, 5)     |    0.8218     |     42.35     |
|      DTC      |       3       |    (4, 4)     |    (5, 5)     |    0.8752     |     52.48     |
| Random Forest |       3       |    (4, 4)     |    (5, 5)     |    0.9675     |     52.55     |
|      SVM      |       4       |    (2, 2)     |    (1, 1)     |    0.7029     |    405.15     |
|      DTC      |       4       |    (2, 2)     |    (1, 1)     |    0.8172     |    390.41     |
...
|      SVM      |       4       |    (3, 3)     |    (5, 5)     |    0.8987     |     83.57     |
|      DTC      |       4       |    (3, 3)     |    (5, 5)     |    0.8210     |    127.07     |
| Random Forest |       4       |    (3, 3)     |    (5, 5)     |    0.9568     |    101.53     |
|      SVM      |       4       |    (4, 4)     |    (1, 1)     |    0.7379     |    177.68     |
|      DTC      |       4       |    (4, 4)     |    (1, 1)     |    0.8471     |    106.48     |
| Random Forest |       4       |    (4, 4)     |    (1, 1)     |    0.9598     |    110.33     |
|      SVM      |       4       |    (4, 4)     |    (2, 2)     |    0.8699     |    286.01     |
|      DTC      |       4       |    (4, 4)     |    (2, 2)     |    0.8628     |    108.82     |
| Random Forest |       4       |    (4, 4)     |    (2, 2)     |    0.9644     |    110.05     |
|      SVM      |       4       |    (4, 4)     |    (3, 3)     |    0.9112     |    176.43     |
|      DTC      |       4       |    (4, 4)     |    (3, 3)     |    0.8669     |     99.09     |
| Random Forest |       4       |    (4, 4)     |    (3, 3)     |    0.9665     |     92.66     |
|      SVM      |       4       |    (4, 4)     |    (4, 4)     |    0.8802     |     60.62     |
|      DTC      |       4       |    (4, 4)     |    (4, 4)     |    0.8699     |     80.38     |
| Random Forest |       4       |    (4, 4)     |    (4, 4)     |    0.9656     |     74.32     |
|      SVM      |       4       |    (4, 4)     |    (5, 5)     |    0.8519     |     44.45     |
|      DTC      |       4       |    (4, 4)     |    (5, 5)     |    0.8699     |     59.10     |
| Random Forest |       4       |    (4, 4)     |    (5, 5)     |    0.9665     |     56.08     |
|      SVM      |       6       |    (2, 2)     |    (1, 1)     |    0.7607     |    406.91     |
|      DTC      |       6       |    (2, 2)     |    (1, 1)     |    0.8080     |    395.91     |
| Random Forest |       6       |    (2, 2)     |    (1, 1)     |    0.9587     |    395.01     |
|      SVM      |       6       |    (2, 2)     |    (2, 2)     |    0.8628     |    434.92     |
|      DTC      |       6       |    (2, 2)     |    (2, 2)     |    0.8203     |    455.43     |
| Random Forest |       6       |    (2, 2)     |    (2, 2)     |    0.9616     |    424.93     |
|      SVM      |       6       |    (2, 2)     |    (3, 3)     |    0.8795     |    431.59     |
|      DTC      |       6       |    (2, 2)     |    (3, 3)     |    0.8253     |    479.00     |
| Random Forest |       6       |    (2, 2)     |    (3, 3)     |    0.9636     |    389.71     |
|      SVM      |       6       |    (2, 2)     |    (4, 4)     |    0.8942     |    439.47     |
|      DTC      |       6       |    (2, 2)     |    (4, 4)     |    0.8273     |    529.83     |
| Random Forest |       6       |    (2, 2)     |    (4, 4)     |    0.9651     |    356.95     |
...
| Random Forest |       6       |    (3, 3)     |    (3, 3)     |    0.9553     |    150.49     |
|      SVM      |       6       |    (3, 3)     |    (4, 4)     |    0.9193     |    188.65     |
|      DTC      |       6       |    (3, 3)     |    (4, 4)     |    0.8005     |    166.91     |
| Random Forest |       6       |    (3, 3)     |    (4, 4)     |    0.9564     |    127.38     |
|      SVM      |       6       |    (3, 3)     |    (5, 5)     |    0.9088     |     89.53     |
|      DTC      |       6       |    (3, 3)     |    (5, 5)     |    0.8069     |    152.78     |
| Random Forest |       6       |    (3, 3)     |    (5, 5)     |    0.9579     |    108.62     |
|      SVM      |       6       |    (4, 4)     |    (1, 1)     |    0.8053     |    184.38     |
|      DTC      |       6       |    (4, 4)     |    (1, 1)     |    0.8509     |    107.66     |
| Random Forest |       6       |    (4, 4)     |    (1, 1)     |    0.9621     |    111.45     |
|      SVM      |       6       |    (4, 4)     |    (2, 2)     |    0.8918     |    289.62     |
|      DTC      |       6       |    (4, 4)     |    (2, 2)     |    0.8518     |    116.97     |
| Random Forest |       6       |    (4, 4)     |    (2, 2)     |    0.9700     |    111.78     |
|      SVM      |       6       |    (4, 4)     |    (3, 3)     |    0.9204     |    188.93     |
|      DTC      |       6       |    (4, 4)     |    (3, 3)     |    0.8541     |    112.89     |
| Random Forest |       6       |    (4, 4)     |    (3, 3)     |    0.9683     |     95.93     |
|      SVM      |       6       |    (4, 4)     |    (4, 4)     |    0.8986     |     63.41     |
|      DTC      |       6       |    (4, 4)     |    (4, 4)     |    0.8573     |     99.39     |
... best result:
|__Random Forest__|  __6__      |  __(4, 4)__   |  __(4, 4)__   |  __0.9701__   |   __79.80__   |
|      SVM      |       6       |    (4, 4)     |    (5, 5)     |    0.8772     |     47.23     |
|      DTC      |       6       |    (4, 4)     |    (5, 5)     |    0.8548     |     72.41     |
| Random Forest |       6       |    (4, 4)     |    (5, 5)     |    0.9695     |     60.55     |
|      SVM      |      18       |    (2, 2)     |    (1, 1)     |    0.8022     |    415.57     |
|      DTC      |      18       |    (2, 2)     |    (1, 1)     |    0.7768     |    419.54     |
| Random Forest |      18       |    (2, 2)     |    (1, 1)     |    0.9438     |    405.65     |
|      SVM      |      18       |    (2, 2)     |    (2, 2)     |    0.8718     |    460.13     |
|      DTC      |      18       |    (2, 2)     |    (2, 2)     |    0.7877     |    553.30     |
| Random Forest |      18       |    (2, 2)     |    (2, 2)     |    0.9487     |    450.64     |
|      SVM      |      18       |    (2, 2)     |    (3, 3)     |    0.8870     |    529.68     |
|      DTC      |      18       |    (2, 2)     |    (3, 3)     |    0.7903     |    780.02     |
| Random Forest |      18       |    (2, 2)     |    (3, 3)     |    0.9496     |    463.83     |
|      SVM      |      18       |    (2, 2)     |    (4, 4)     |    0.9003     |    618.61     |
|      DTC      |      18       |    (2, 2)     |    (4, 4)     |    0.7918     |    1003.52    |
| Random Forest |      18       |    (2, 2)     |    (4, 4)     |    0.9510     |    499.87     |
|      SVM      |      18       |    (2, 2)     |    (5, 5)     |    0.9117     |    722.95     |
|      DTC      |      18       |    (2, 2)     |    (5, 5)     |    0.7938     |    1370.02    |
| Random Forest |      18       |    (2, 2)     |    (5, 5)     |    0.9524     |    480.86     |
|      SVM      |      18       |    (3, 3)     |    (1, 1)     |    0.8161     |    231.41     |
|      DTC      |      18       |    (3, 3)     |    (1, 1)     |    0.7543     |    180.59     |
| Random Forest |      18       |    (3, 3)     |    (1, 1)     |    0.9317     |    178.47     |
|      SVM      |      18       |    (3, 3)     |    (2, 2)     |    0.9000     |    269.27     |
|      DTC      |      18       |    (3, 3)     |    (2, 2)     |    0.7671     |    216.17     |
| Random Forest |      18       |    (3, 3)     |    (2, 2)     |    0.9397     |    186.23     |
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
Summary totalPass each: 	90 (many lines ommited in above table for brevity, see file: output.txt for complete result)

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
