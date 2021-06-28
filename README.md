# Classifier Comparison
The objective of this assignment are the following: 

    - Use/implement a feature selection/reduction technique.
    - Experiment with various classification models.
    - Think about dealing with imbalanced data. 
    
## Data Description:

The dataset is split into training and test sets; both files are in CSV format. The training dataset consists of 14,528 records and the test dataset consists of 116,205 records. We provide you the class labels in the training set, and the test labels are held out. There are 54 attributes in each of the training and test sets; the training set has an additional attribute for class labels (the last column). 

Attributes 1-54 are numeric cartographic variables – some of them are binary variables indicating absence or presence of something, such as a particular soil type. Specifically, attributes #1, 8, 9, 20, 22, 31, 42, 47, 50, 54 are numeric, and the rest are all binary (except the one for class labels).

    train.csv: Training set with 14,528 records (each row is a record). Each record contains 55 attributes. The last attribute is the class label (1~7).

    test_no_label.csv: Test set with 116,205 records (each row is a record). Each record contains 54 attributes since the class labels are withheld.

    format.dat: A sample submission with 116,205 entries of randomly chosen numbers between 1 and 7. 
