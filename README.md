# Mini Project - 3

## Learning to add digits

Project 3 for Comp 551, Fall 16 at McGill University

**Kaggle Team** : Xenos

## Usage

### Logistic Regression Implemention

* Logistic regression with cross-validation is performed using the LogisticRegressionCV method from scikit-learn.
* _logisticRegression.py_ assumes that a Dataset directory containing `train_x` and `train_y` is placed at the same level within the directory structure. Change the values in main for inputs and examples to modify the train/test split. If you have previously trained a model, the package will first look for a pickled file of the preprocessed images and the trained logistic regressor.
* To run, call `python logisticRegression.py`

### Neural Network Implementation

* Make sure that the path to the input files are accurately set as in the neural_net.py
* Run the file _neural_net.py_ : `python neural_net.py`

### Convolutional Neural Network Implementation

* Installation prerequisites : Theano (latest) and Lasagne (latest)
* Run the file _runner_lasagne.py_. Usage :

```
usage: runner_lasagne.py [-h] -e EPOCHS [-m MINI] [-s START]

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS Number of epochs
  -m MINI, --mini MINI  Number of minibatches
  -s START, --start START Starting weights
```

## Authors

* Koustuv Sinha. McGill ID : 260721248, _koustuv.sinha@mail.mcgill.ca_
* Caitrin Armstrong, McGill ID : 260501112, _caitrin.armstrong@mail.mcgill.ca_
* Shabir Abdul Shamadh, McGill ID : 260723366, _shabir.abdulsamadh@mail.mcgill.ca_
