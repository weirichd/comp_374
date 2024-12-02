# COMP 374 - Deep Learning


Below is a first attempt at a full course on deep learning.
This is still a draft and is subject to revision.


## Core concepts covered in the class

* Neural Network Archetrcture
* Tensorflow and Keras
* Training Neural Networks
* Loss functions
* Gradient Descent & Back Propogation
* Data preprocessing
* Convolutional Neural Networks
* Applications of CNNs
* RNNs (??)
* Transformer Models and Attention
* Evaluation and Interpretation of Models
* Limitations and Challenges
* Demo and Exam



## Weekly Schedule

Proposed outline for course

### Week 1 - Core Mathematical Concepts and Background

Introduction to foundational concepts
* What is a derivative?
* What is a partial derivative?
* What are vectors and matrices?
* What is curve fitting?
* Machine Learning as a Method to Make Future Predictions from Past Data
* Two non-deep learning foundational examples
    * Linear regression for continuous targets
    * Logistic regression for classification
* Overview of Python and Jupyter Environments

#### Key Vocab:

* Derivative
* Matrix
* Vector
* Classification
* Regression
* Feature/Independant Variable
* Target/Dependant Variable
* Machine Learning

### Week 2 - Introduction to Neural Networks

Introduction to the kind of problem we are interested in solving
* General-purpose curve fitting requirements
    * A function which is robust enough to fit any data we throw at it
    * A training algorithm for this function which conforms to physical constraints of a computer
* Why are neural networks a good solution?
* Neural networks as mathematical functions
* Deep vs. Shallow networks
* First look at Tensorflow and Keras


#### Key Vocab:

* Neural Network
* Deep Learning
* Feed-Forward Neural Network
* Weights/Bias
* Activation Function 
* Layer
* Universal Approximation Theorem
* Keras
    * Sequential Model
    * Dense Layers
    * Model Compilation

### Week 3 - Training

Now that we have the shape of our general purpose curve fitting function, how do we train them?

* What is training?
* Gradient descent
* Stochastic Gradient descent
* Train/test split
* Learning curves
* Hyperparameters 
    * Numebr of layers
    * Size of each layer
* Dropout
* L1/L2 regularization    
* Train our first network with Keras

#### Key Vocab:

* Training
* Loss functions
* Hyperparameter
* Regularization
* Model.fit()

### Week 4 - Convolutional Neural Networks

* What is convolution in general?
* What is a convolutional layer?
* If Feed-forward neural networks are genral, why do we even need CNNs?
* Build a NN for image classification using a FFNN and then a CNN
* Brief discussion of recurrant neural networks
    * Another example of a specialized network structure to handle a specific kind of data
    * Recently fallen out of favor after invention of transformers (discussed in 2nd half of semester)
 
#### Key Vocab:

* Neural Network Architecture
* Convolution
* Tensors
* Pooling Layer


### Week 5 - Generative Models

What are generative models

* "Generation" really just means regression w/ a specific kind of target
* Autoencoders
* Basics on Diffusion techniques

#### Key Vocab:

* Generation
* Autoencoder

### Week 6

Midterm?

#### Key Vocab:

### Week 7

Introduction to Transformers


#### Key Vocab:


### Week 8

#### Key Vocab:

### Week 9
#### Key Vocab:
### Week 10
#### Key Vocab:
### Week 11
#### Key Vocab:
### Week 12
#### Key Vocab:


