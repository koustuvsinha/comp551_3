import csv
import math
import numpy as np
from PIL import Image

"""
    function to load the given training set of data.
        num_of_examples - number of total data samples available in the data set.
        :returns - an object consisting of all the training data read from file
"""


def load_train_set(num_of_examples):
    x = np.fromfile('../Dataset/train_x.bin', dtype='uint8')  # load the data from file
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
    with open('../Dataset/train_y.csv', newline='') as csvReader:  # Reads the given csv
        train_y = csv.reader(csvReader)
        for output in train_y:
            if not output[1] == 'Prediction':  # skip the first header row in the dataset
                y_list.append(output[1])

    y_list = np.asarray(y_list)  # convert loaded output labels into an array
    return y_list


"""
    function to do PCA dimensionality reduction and do whitening on the data to make it a gaussian with 0 mean
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


"""
    function to train a Neural Network of two hidden layers.
        X           - the image train data after all pre-processing is done
        y           - output labels of the data
        h           - no of nodes in the first hidden layer
        h2          - no of nodes in the second hidden layer
        outs        - no of output nodes in the NN
        step_size   - the step size to be taken for gradient descent
        reg         - the regularization strength of the model
        dim         - no of available dimensions per image
        iterations  - no of iterations to do learning and back-porpagation
        :returns
            lost_list         - the list of the "loss" per iteration
            best_param_list   - the params width he least loss value over all iterations
            final_params      - the final model parameters after the learning is over
"""


def two_hiddenLayer_NN(X, y, h, h2, outs, step_size, reg, dim, iterations):
    # initialize parameters randomly
    W = 0.01 * np.random.randn(dim, h)
    W = W / np.sqrt(2.0 / dim)
    b = np.zeros((1, h))

    W2 = 0.01 * np.random.randn(h, h2)
    W2 = W2 / np.sqrt(2.0 / h)
    b2 = np.zeros((1, h2))

    W3 = 0.01 * np.random.randn(h2, outs)
    W3 = W3 / np.sqrt(2.0 / h2)
    b3 = np.zeros((1, outs))

    num_examples = X.shape[0]
    best_param_list = [1.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    lost_list = []
    loss = 0.0

    # gradient descent loop
    for i in range(iterations):
        #   evaluate class scores, [N x outs]
        hidden_layer_1 = np.maximum(0, np.dot(X, W) + b)  # ReLU activation
        hidden_layer_2 = np.maximum(0, np.dot(hidden_layer_1, W2) + b2)
        scores = np.dot(hidden_layer_2, W3) + b3

        #    compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x outs]

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2) + 0.5 * reg * np.sum(W3 * W3)

        # store previous loss; incase of termination condition is reached
        loss_prev = loss
        loss = data_loss + reg_loss

        # termination condition for the learning. terminate if the loss reaches 'NaN' or 'Infinity'
        if math.isnan(loss) or loss == float('Inf'):
            print("Termination Condition for 'Loss' reached at iteration %d" % i)
            loss = loss_prev
            break

        lost_list.append(loss)
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))

        if loss < best_param_list[0]:
            best_param_list[0] = loss
            best_param_list[1] = W
            best_param_list[2] = b
            best_param_list[3] = W2
            best_param_list[4] = b2
            best_param_list[5] = W3
            best_param_list[6] = b3

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # back propagation on the second hidden layer parameter
        dW3 = np.dot(hidden_layer_2.T, dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)
        dhidden_2 = np.dot(dscores, W3.T)

        # for the ReLU function [dr/dx = 1(x>0)]
        dhidden_2[hidden_layer_2 <= 0] = 0

        # back propagation on the first hidden layer parameter
        dW2 = np.dot(hidden_layer_1.T, dhidden_2)
        db2 = np.sum(dhidden_2, axis=0, keepdims=True)
        dhidden_1 = np.dot(dhidden_2, W2.T)

        # for the ReLU function [dr/dx = 1(x>0)]
        dhidden_1[hidden_layer_1 <= 0] = 0

        # finally the first layer parameters W, X
        dW = np.dot(X.T, dhidden_1)
        db = np.sum(dhidden_1, axis=0, keepdims=True)

        # regularization gradient
        dW3 += reg * W3
        dW2 += reg * W2
        dW += reg * W

        # store params from previous iteration before update
        #       this is done incase the termination condition becomes TRUE after update
        #       and we exit the learning phase, we need the last best parameters.
        W_prev = W
        b_prev = b
        W2_prev = W2
        b2_prev = b2
        W3_prev = W3
        b3_prev = b3

        # perform a parameter update on the weights and the biases
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2
        W3 += -step_size * dW3
        b3 += -step_size * db3

        # check if termination condition reached upon parameter update
        #            if so stop learning and store the params of the previous iteration as best parameters.
        if ( \
            (W == float('inf')).any() or np.isnan(W).any() or \
            (b == float('inf')).any() or np.isnan(b).any() or \
            (W2 == float('inf')).any() or np.isnan(W2).any() or \
            (b2 == float('inf')).any() or np.isnan(b2).any() or \
            (W3 == float('inf')).any() or np.isnan(W3).any() or \
            (b3 == float('inf')).any() or np.isnan(b3).any()
        ):
            print("Termination Condition reached at iteration %d" % i)
            W = W_prev
            b = b_prev
            W2 = W2_prev
            b2 = b2_prev
            W3 = W3_prev
            b3 = b3_prev
            break

    final_params = [loss, W, b, W2, b2, W3, b3]
    return lost_list, best_param_list, final_params


"""
    function to train a Neural Network of a single hidden layer.
        X           - train data after all pre-processing is done
        y           - output labels of the data
        h           - no of nodes in the hidden layer
        outs        - no of output nodes in the NN
        step_size   - the step size to be taken for gradient descent
        reg         - the regularization strength of the model
        dim         - no of available dimensions per image
        iterations  - no of iterations to do learning and back-porpagation
        :returns
            lost_list         - the list of the "loss" per iteration
            best_param_list   - the params width he least loss value over all iterations
            final_params      - the final model parameters after the learning is over
"""


def one_hiddenLayer_NN(X, y, h, outs, step_size, reg, dim, iterations):
    # initialize parameters randomly
    W = 0.01 * np.random.randn(dim, h)
    W = W / np.sqrt(2.0 / dim)
    b = np.zeros((1, h))

    W2 = 0.01 * np.random.randn(h, outs)
    W2 = W2 / np.sqrt(2.0 / h)
    b2 = np.zeros((1, outs))

    num_examples = X.shape[0]
    best_param_list = [1.00, 0.0, 0.0, 0.0, 0.0]
    lost_list = []
    loss = 0.0

    # gradient descent loop
    for i in range(iterations):
        #   evaluate class scores, [N x outs]
        hidden_layer = np.maximum(0, np.dot(X, W) + b)  # ReLU activation
        scores = np.dot(hidden_layer, W2) + b2

        #    compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x outs]

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)

        # store previous loss; incase of termination condition is reached
        loss_prev = loss
        loss = data_loss + reg_loss

        # termination condition for the learning.
        # terminate if the loss reaches 'NaN' or 'Infinity'
        if math.isnan(loss) or loss == float('Inf'):
            print("Termination Condition for 'Loss' reached at iteration %d" % i)
            loss = loss_prev
            break

        lost_list.append(loss)
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))

        if loss < best_param_list[0]:
            best_param_list[0] = loss
            best_param_list[1] = W
            best_param_list[2] = b
            best_param_list[3] = W2
            best_param_list[4] = b2

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # back propagation on the hidden layer parameters W2 b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        dhidden = np.dot(dscores, W2.T)

        # for the ReLU function [dr/dx = 1(x>0)]
        dhidden[hidden_layer <= 0] = 0

        # finally the first layer parameters W, X
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        # regularization gradient
        dW2 += reg * W2
        dW += reg * W

        # store params from previous iteration before update
        #       this is done incase the termination condition becomes TRUE after update
        #       and we exit the learning phase, we need the last best parameters.
        W_prev = W
        b_prev = b
        W2_prev = W2
        b2_prev = b2

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

        # check if termination condition reached upon parameter update
        #            if so stop learning and store the params of the previous iteration as best parameters.
        if ( \
                (W == float('inf')).any() or np.isnan(W).any() or \
                (b == float('inf')).any() or np.isnan(b).any() or \
                (W2 == float('inf')).any() or np.isnan(W2).any() or \
                (b2 == float('inf')).any() or np.isnan(b2).any() \
            ):
            print("Termination Condition reached at iteration %d" % i)
            W = W_prev
            b = b_prev
            W2 = W2_prev
            b2 = b2_prev
            break

    final_params = [loss, W, b, W2, b2]
    return lost_list, best_param_list, final_params


"""
    function to get the "training accuracy" of the learned model
        X                   - the input data used for training the model
        y                   - the output labels of the data used for training
        no_hidden_layers    - which training NN model was used (one with 2 hidden layers or 1)
        final_params        - the final learned parameters of the model (Weights + biases)
        :returns            - the train accuracy of the model
"""


def get_accuracy(X, y, no_hidden_layers, final_params):
    # get the final learned model parameters to validate accuracy and
    # checking training accuracy
    if no_hidden_layers == 1:
        W = final_params[1]
        b = final_params[2]
        W2 = final_params[3]
        b2 = final_params[4]

        hidden_layer = np.maximum(0, np.dot(X, W) + b)
        scores = np.dot(hidden_layer, W2) + b2
        predicted_class = np.argmax(scores, axis=1)

    elif no_hidden_layers == 2:
        W = final_params[1]
        b = final_params[2]
        W2 = final_params[3]
        b2 = final_params[4]
        W3 = final_params[5]
        b3 = final_params[6]

        hidden_layer_1 = np.maximum(0, np.dot(X, W) + b)
        hidden_layer_2 = np.maximum(0, np.dot(hidden_layer_1, W2) + b2)
        scores = np.dot(hidden_layer_2, W3) + b3
        predicted_class = np.argmax(scores, axis=1)

    else:
        print("No of hidden_layers should be either 1 or 2")
        return None
    return (np.mean(predicted_class == y))


"""
    function that splits the given dataset into K-folds for cross validation.
        X                - the entire input set
        y                - all the output labels per each input sample
        dims             - no of dimensions per sample to be considered when doing PCA
        iter_per_fold    - iterations per fold in K-fold cross validation
        K                - number of folds to be considered in cross-validation
        :returns         - a list of parameter and output information of each fold
"""


def K_fold_crossV(X_full, y_full, dims, iter_per_fold, K):
    fold_info = []
    data_per_fold = int(X_full.shape[0] / K)

    for fold in range(0, K):
        print("Training on fold - ", fold + 1)
        strt_idx = fold * data_per_fold
        end_idx = strt_idx + data_per_fold

        X_lft_out = X_full[strt_idx:end_idx, ]
        y_lft_out = y_full[strt_idx:end_idx, ]

        if (strt_idx == 0):
            X = X_full[end_idx:, ]
            y = y_full[end_idx:, ]
        else:
            X = X_full[0:strt_idx, ]
            X_2 = X_full[end_idx:, ]
            X = np.concatenate((X, X_2), axis=0)

            y = y_full[0:strt_idx, ]
            y_2 = y_full[end_idx:, ]
            y = np.concatenate((y, y_2), axis=0)

        # PCA reduction and whitening of the images
        X = preprocess_images(X, dims)

        # 2 Hidden layer NN
        loss, best_params, final_params = two_hiddenLayer_NN(X, y, 50, 50, 19, 1e-0, 1e-3, dims, iter_per_fold)
        print('Final train loss: %.2f' % final_params[0])

        # 1 Hidden layer NN
        # loss, best_params, final_params = one_hiddenLayer_NN(X, y, 5, 19, 1e-0 , 1e-3, dims, iter_per_fold)
        # print('Final train loss: %.2f' % final_params[0])

        train_accuracy = get_accuracy(X, y, 2, final_params)
        # train_accuracy = get_accuracy(X, y, 1, final_params)
        print('Training accuracy: %.2f' % train_accuracy)

        # PCA reduction and whitening of the images of the test set
        X_lft_out = preprocess_images(X_lft_out, dims)

        test_accuracy = get_accuracy(X_lft_out, y_lft_out, 2, final_params)
        # test_accuracy = get_accuracy(X_lft_out, y_lft_out, 1, final_params)
        print('Testing accuracy: %.2f' % test_accuracy)

        newInfo = []
        newInfo.append(fold)
        newInfo.append(loss)
        newInfo.append(train_accuracy)
        newInfo.append(test_accuracy)
        newInfo.append(final_params)
        newInfo.append(best_params)

        fold_info.append(newInfo)
        print("---------------------------------------")

    return fold_info


"""
    Main execution of the code base
"""

if __name__ == '__main__':
    num_of_imgs = 100000  # total number of images in the train_data
    no_of_eggs = 75000  # total number of images to use for training the NN
    dims = 500  # define the number of dimensions out of the total 3600 to consider in training
    outs = 19  # no of output nodes in the NN
    iterations = 10000

    x = load_train_set(num_of_imgs)  # load the dataset
    Xt = load_images_to_array(x, num_of_imgs)  # load image pixels into array

    X = Xt.T  # Xt - [3600 x 100000] transpose it to have image pixel data per row
    # X = X[:no_of_eggs, :]       # separate only the no of images to be considered for training

    y_list = load_output_labels()  # load the output labels
    y = np.zeros(num_of_imgs, dtype='uint8')
    # separate only the output labels of the images considered for training
    for item in range(0, len(y)):
        y[item] = y_list[item]

    # PCA reduction and whitening of the images
    # X = preprocess_images(X, dims)

    folds_info = K_fold_crossV(X, y, dims, iterations, 5)

    # train the data to get the NN model
    # 2 Hidden layer NN
    # loss, best_params, final_params = two_hiddenLayer_NN(X, y, 50, 50, 19, 1e-0 , 1e-3, dims, 20000)

    # 1 Hidden layer NN
    # loss, best_params, final_params = one_hiddenLayer_NN(X, y, 100, 19, 1e-0 , 1e-3, dims, 20000)

    # train_accuracy = get_accuracy(X, y, 2, final_params)
    # print('Training accuracy: %.2f' % train_accuracy)
