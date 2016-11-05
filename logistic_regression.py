import csv
import numpy as np
from PIL import Image
import sklearn as sk
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
import argparse

def main():   
    num_of_imgs = 100000  # total number of images in the train_data
    no_of_eggs = 1000
    dims = 500
    
    x = load_train_set(num_of_imgs)  # load the dataset
    Xt = load_images_to_array(x, num_of_imgs)  # load image pixels into array

    X = Xt.T  # Xt - [3600 x 100000] transpose it to have image pixel data per row
    X = X[:no_of_eggs, :]       # separate only the no of images to be considered for training

    print "loaded and sliced data"
    X = preprocess_images(X, dims)

    print "preprocessed images"
    
    y_list = load_output_labels()  # load the output labels
    y = np.zeros(num_of_imgs, dtype='uint8')
    # separate only the output labels of the images considered for training
    for item in range(0, len(y)):
        y[item] = y_list[item]

    y = y[:no_of_eggs]

    print "loaded and sliced y"
    
    logreg = LogisticRegressionCV(cv=10,solver='sag',n_jobs=-1,verbose=1,multi_class='multinomial')
    logreg.fit(X, y)

    pickle.dump(logreg, open("logreg.p", "wb"))
    
    x_test = load_test_set(num_of_imgs)  # load the dataset
    Xt = load_images_to_array(x_test, num_of_imgs)  # load image pixels into array

    X_test = Xt.T  # Xt - [3600 x 100000] transpose it to have image pixel data per row
    X_test = X_test[:no_of_eggs, :]       # separate only the no of images to be considered for training

    X_test = preprocess_images(X_test, dims)

    
    # train the data to get the NN model
    # 2 Hidden layer NN
    # loss, best_params, final_params = two_hiddenLayer_NN(X, y, 50, 50, 19, 1e-0 , 1e-3, dims, 20000)

    # 1 Hidden layer NN
    # loss, best_params, final_params = one_hiddenLayer_NN(X, y, 100, 19, 1e-0 , 1e-3, dims, 20000)

    # train_accuracy = get_accuracy(X, y, 2, final_params)
    # print('Training accuracy: %.2f' % train_accuracy)    pass

"""
    function to load the given training set of data.
        num_of_examples - number of total data samples available in the data set.
        :returns - an object consisting of all the training data read from file
"""


def load_train_set(num_of_examples):
    x = np.fromfile('Dataset/train_x.bin', dtype='uint8')  # load the data from file
    x = x.reshape((num_of_examples, 60, 60))  # reshape the read data into a manipulative format
    return x

def load_test_set(num_of_examples):
    x = np.fromfile('../Dataset/test_x.bin', dtype='uint8')  # load the data from file
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

    cov = np.dot(X.T, X) / X.shape[0]  # get the covariance matrix of X
    U, S, V = np.linalg.svd(cov)  # S,V,D factorization of the covariance matrix (U - eigenvectors)
    Xrot = np.dot(X, U)  # decorrelate the data by projecting the original data into its eigenbasis

    Xrot_reduced = np.dot(Xrot, U[:, :dimensions])  # dimensionality reduction
    S_reduced = S[:dimensions]

    # do whitening on the data: divide by the eigenvalues (which are square roots of the singular values)
    Xwhite = Xrot_reduced / np.sqrt(S_reduced + 1e-5)
    return Xwhite



if __name__ == "__main__":
    main()

 
