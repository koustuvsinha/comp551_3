import csv
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
import argparse
import pickle

'''
main method running LogisticRegressionCV. Thanks to Shabir Abdul Samadh for the image
processing methods
'''
def main():   
    num_of_imgs = 100000  # total number of images in the train_data
    no_of_eggs = 80000
    no_of_test= 20000
    dims = 3600
    
    try:
        X = pickle.load(open("preprocessed_images_3600.p", "rb"))
    except:
        print "in exception"
        x = load_train_set(num_of_imgs)  # load the dataset
        Xt = load_images_to_array(x, num_of_imgs)  # load image pixels into array
        print "in exception, loaded images"
        X = Xt.T
        X = preprocess_images(X, dims)
        print "in exception, processed images"
        file_name = "preprocessed_images_3600.p"
        pickle.dump(X, open(file_name, "wb"))
        print "in exception, dumped images"
    
    X_train = X[:no_of_eggs, :]# separate only the no of images to be considered for training

    print "loaded and preprocessed images"
    
    y_list = load_output_labels()
    y = np.zeros(num_of_imgs, dtype='uint8')
    for item in range(0, len(y)):
        y[item] = y_list[item]

    y_train = y[:no_of_eggs]

    print "loaded and sliced y"
    
    classifier_name = "logreg_" + str(no_of_eggs) + "_" + str(no_of_test) + ".p"
    try:
        logreg = pickle.load(open(classifier_name, "rb"))
    except:
        logreg = LogisticRegressionCV(cv=10,solver='sag',n_jobs=-1,verbose=1,multi_class='multinomial')
        logreg.fit(X, y)
        pickle.dump(logreg, open(classifier_name, "wb"))

    
    print "fit classifier"
    print "scores", logreg.scores_
    print "C", logreg.C_
    print "iter", logreg.n_iter_
    
    test_boundary = no_of_eggs + no_of_test
    
    X_test = X[no_of_eggs:test_boundary, :]
    print "sliced out test data"
    y_test = y[no_of_eggs:test_boundary]
    pred = logreg.predict(X_test)
    print classification_report(y_test, pred)

"""
    function to load the given training set of data.
        num_of_examples - number of total data samples available in the data set.
        :returns - an object consisting of all the training data read from file
"""
def load_train_set(num_of_examples):
    x = np.fromfile('Dataset/train_x.bin', dtype='uint8')  # load the data from file
    x = x.reshape((num_of_examples, 60, 60))  # reshape the read data into a manipulative format
    return x


"""
    function to initalize a 2D array of image pixels from the loaded dataset
        data - the loaded data object of all training data
        total_no_images - total number of images in the loaded data object
        :returns - a 2D array with image pixel imformation in a flattened vector
"""
def load_images_to_array(data, total_no_images):
    # initialize an array to hold only the part of the data to be used for training
    Xt = np.zeros((3600, total_no_images))
    # separately set up the no of data samples to be considered for training
    for image in range(total_no_images):
        img = Image.fromarray(data[image])
        pix = np.asarray(img).flatten()
        Xt[:, image] = pix
    return Xt


"""
    function that loads the output labels of the given data set from file
        :returns - the output labels of the given data set loaded as a flat array.
"""


def load_output_labels():
    y_list = []
    with open('Dataset/train_y.csv') as csvReader:  # Reads the given csv
        train_y = csv.reader(csvReader)
        for output in train_y:
            if not output[1] == 'Prediction':  # skip the first header row in the dataset
                y_list.append(output[1])

    y_list = np.asarray(y_list)  # convert loaded output labels into an array
    return y_list


"""
    function that loads the output labels of the given data set from file
        :returns - the output labels of the given data set loaded as a flat array.
"""


def load_output_labels_test():
    y_list = []
    with open('Dataset/test_y.csv', newline='') as csvReader:  # Reads the given csv
        test_y = csv.reader(csvReader)
        for output in test_y:
            if not output[1] == 'Prediction':  # skip the first header row in the dataset
                y_list.append(output[1])

    y_list = np.asarray(y_list)  # convert loaded output labels into an array
    return y_list


"""    function to do PCA dimensionality reduction and do whitening on the data to make it a gaussian with 0 mean
    and identity covariance matrix
        X           - 2D array of all training images with image-pixel data per row
        dimensions  - the no of dimensions to which the data is to be reduced
        :returns    - a 2D array of training data with PCA transformation and whitening.
"""


def preprocess_images(X, dimensions):
    X -= np.mean(X, axis=0)  # Mean subtraction to center the data around the same origin
    # PCA to reduce the total number of dimensions to only the most contributing ones
    print "mean"
    cov = np.dot(X.T, X) / X.shape[0]  # get the covariance matrix of X
    print "cov"
    U, S, V = np.linalg.svd(cov)  # S,V,D factorization of the covariance matrix (U - eigenvectors)
    print "factorization"
    Xrot = np.dot(X, U)  # decorrelate the data by projecting the original data into its eigenbasis
    print "decorrelate"
    Xrot_reduced = np.dot(Xrot, U[:, :dimensions])  # dimensionality reduction
    S_reduced = S[:dimensions]
    print "reduce dimensions"
    # do whitening on the data: divide by the eigenvalues (which are square roots of the singular values)
    Xwhite = Xrot_reduced / np.sqrt(S_reduced + 1e-5)
    return Xwhite



if __name__ == "__main__":
    main()

 
