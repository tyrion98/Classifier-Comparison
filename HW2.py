# import needed libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import time
# scaler
from sklearn.preprocessing import StandardScaler
# import knn
from sklearn.neighbors import KNeighborsClassifier
# import knn classifier
from sklearn.tree import DecisionTreeClassifier
# import naive bayes classifier
# using the bernoulli version
from sklearn.naive_bayes import *
start_time = time.time()


def set_up():
    # initialize list of attributes
    n = []
    # has 54 cartographic variables
    [n.append(i) for i in range(1,55)]
    # last one is the class variable (valid vars are from 1-7)
    n += ['class']
    train_set = pd.read_csv('train.csv', names=n)
    # for the test set
    n2 = []
    # has 54 cartographic variables
    [n2.append(i) for i in range(1,55)]
    test_set = pd.read_csv('test_no_label.csv', names=n2)

    # print(train_set.head())
 
    # split dataset into features and labels
    # x -> contains first 54 columns of dataset (features)
    X = train_set.iloc[:, :-1].values
    # y -> contains the (labels)
    y = train_set.iloc[:, 54].values


    # converts test data into a list
    X_test = test_set.values

    # split training set by test size and  (CROSS_VAL)
    my_train_split(X,y)

    # predict final test set with the best classifier found from our validation function
    normalize_predict(X, y, X_test)

    # Notify the user running the program that its finished predicting the class labels for the dataset
    print("Test Predictions have been added to the test_predictions.dat file")
    print("Program finished running in ", time.time() - start_time, "seconds")
    return


# my version of the train test split function
def my_train_split(X, y, test_size = 0.30):    
    # X_test and y_test are basically the class labels
    # X_train and y_train are like everything else
    # split into test size percentage
    test_length = int(len(X) *test_size)
    # leftover will be the training data
    train_length = len(X) - test_length

    # DEFINE TRAIN SET
    # 0 - 4357
    x_train = X[:test_length]
    y_train = y[:test_length]
    # DEFINE TEST SET
    x_test = X[train_length:] # SAME
    # DO SOME NORMALIZING
    scaler = StandardScaler()
    scaler.fit(X)
    # normalizes the features in x train and x test
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_train_update = y[train_length:] # SAME

    # params for second call
    x2_train = X[test_length:]
    y2_train = y[test_length:]
    # define test set
    x2_test = X[:test_length] # SAME
    # normalize
    # normalizes the features in x train and x test
    x2_train = scaler.transform(x2_train)
    x2_test = scaler.transform(x2_test)
    y2_train_update = y[:test_length] # SAME

    # params for third call ( test set starts from middle til test set length rest will be training)
    double_test_val = test_length * 2
    x3_train = X[:test_length] 
    y3_train = y[:test_length]
    temp = X[double_test_val:]
    temp2 = y[double_test_val:]
    x3_train = np.concatenate((x3_train, temp), axis=0)
    y3_train = np.concatenate((y3_train, temp2), axis=0)
    # test = 
    x3_test = X[test_length:double_test_val] # SAME
    # normalize
    # normalizes the features in x train and x test
    x3_train = scaler.transform(x3_train)
    x3_test = scaler.transform(x3_test)
    y3_train_update = y[test_length:double_test_val] # SAME

    # call the classifier calls function 3 times with diff x_train, y_train, x_test, and y_train_update vals
    # first call
    classifier_calls(x_train, y_train, x_test, y_train_update)
    print("-----------------End of First Group--------------------")
    # second call
    classifier_calls(x2_train, y2_train, x2_test, y2_train_update)
    print("-----------------End of Second Group--------------------")
    # third call
    classifier_calls(x3_train, y3_train, x3_test, y3_train_update)
    print("-----------------End of Third Group--------------------")

    # DO THIS LATER
    return 

# use this for all the splits that we will be doing (in this case 3)
def classifier_calls(x_train, y_train, x_test, y_train_update):

    # i put 5 here for clarity but by default its 5 also
    knn_val = do_kNN(x_train, y_train, x_test, 5)
    # knn's prediction
    print("kNN's base prediction: ")
    how_accurate(y_train_update, knn_val)
    # error test here
    # error_test(x_train,y_train,x_test, knn_val) #uncommented because it increases the run time
    # decision tree's prediction
    d_val = do_decision_tree(x_train, y_train, x_test)
    print("decision tree's base prediction: ")
    how_accurate(y_train_update, d_val)
    # error_test(x_train,y_train,x_test, d_val) #uncommented because it increases the run time
    # naive bayes prediction
    nb_val, c = do_naive_bayes(x_train, y_train, x_test)
    print("naive baye's base prediction: ")
    how_accurate(y_train_update, nb_val)
    # error_test(x_train,y_train,x_test, nb_val)  #uncommented because it increases the run time
    # return nothing
    return

# actual running function
def normalize_predict(x_train, y_train, x_test):
    # change later so it picks the best num neighbors from the cross validation function
    # error function did best with num neigh = 5
    num_neigh = 5
    # DO SOME NORMALIZING
    scaler = StandardScaler()
    scaler.fit(x_train)

    # normalizes the features in x train and x test
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # call kNN to get predictions with x num of neighbors
    kNN_y_pred = do_kNN(x_train, y_train, x_test, num_neigh)

    # make files
    file_maker(kNN_y_pred)

    return

# file maker function to use on miner
def file_maker(test_prediction):
    punctuation_ = ['.', ' ', '[', ']']
    f = open("test_prediction.dat", 'a')


    tp = test_prediction.tolist()
    for prediction in tp:
        # if(prediction not in punctuation_):
        #print(prediction)
        f.write(str(prediction))
        f.write('\n')

    # close file
    f.close()
    # returns nothing
    return

# kNN function
def do_kNN(x_train, y_train, x_test, num_neigh):
    # KNN prediction
    k_classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=3, p=1, weights='distance')
    k_classifier.fit(x_train, y_train)
    # using info from training set predict test set val
    kNN_y_pred = k_classifier.predict(x_test)

    # return the prediction
    return kNN_y_pred

# decision tree function
def do_decision_tree(x_train, y_train, x_test):
    d_classifier = DecisionTreeClassifier(class_weight='balanced')
    d_classifier = d_classifier.fit(x_train, y_train)
    d_predict = d_classifier.predict(x_test)

    return d_predict

# naive bayes function
def do_naive_bayes(x_train, y_train, x_test):
    # basic
    bnb = BernoulliNB()
    bnb = bnb.fit(x_train, y_train)
    bnb_pred = bnb.predict(x_test)
    # return the prediction
    return bnb_pred, bnb

# MY OWN ACCURACY SCORE FUNCTION
# shows how accurate each classifier is 
def how_accurate(y_test, y_pred):
    print("Accuracy score:", sep = " ")
    accuracy = 0
    total_correct = 0
    # count num correct
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            total_correct+=1
    # correctly predicted class / total testing class * 100
    accuracy = (total_correct / len(y_test))
    # prints out the accuracy score
    print(accuracy)
    #error_test(X_train, y_train, X_test, y_test)

# helps us pick best k
def error_test(X_train, y_train, X_test, y_test):
    # how to get the best k?
    # try diff values of k and compare them
    error = []
    # Calculating error for K values between 1 and 40
    for i in range(1, 60):
        if i % 2 != 0:
            # makes a classifier with i num neighbors
            knn = KNeighborsClassifier(n_neighbors=i)
            # fits it into the training set
            knn.fit(X_train, y_train)
            # predicts the test set val
            pred_i = knn.predict(X_test)
            # adds the error val to error list
            error.append(np.mean(pred_i != y_test))

    print(error)




# automatically has the program run
set_up()