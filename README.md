# Neural_Network_Charity_Analysis

## Overview

The purpose of this project is to use deep-learning neural networks with the TensorFlow platform in Python, to analyze and classify the success of charitable donations.\
We use the following methods for the analysis:
- preprocessing the data for the neural network model,
- compile, train and evaluate the model,
- optimize the model.

## Resources
- Data Source: [charity_data.csv]
- Software: Python 3.7, Anaconda Navigator 1.9.12, Conda 4.8.4, Jupyter Notebook 6.0.3

## Results

### Deliverable 1:
### Data Preprocessing
- The columns `EIN` and `NAME` are identification information and have been removed from the input data.
- The column `IS_SUCCESSFUL` contains binary data refering to either or not the charity donation was used effectively. This variable is then considered as the target for our deep 
  learning neural network.
- The following columns `APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT` are the features for our model.\
   Encoding of the categorical variables, spliting into training and testing datasets and standardization have been applied to the features.
![d1output.PNG](https://github.com/Praveeja-Sasidharan-Suni/Neural_Network_Charity_Analysis-/blob/main/images/d1output.PNG?raw=true)

### Deliverable 2:
### Compiling, Training, and Evaluating the Model
- This deep-learning neural network model is made of two hidden layers with 100 and 50 neurons respectively.\
The input data has 43 features and 25,724 samples.\
The output layer is made of a unique neuron as it is a binary classification.\
To speed up the training process, we are using the activation function `ReLU` for the hidden layers. As our output is a binary classification, `Sigmoid` is used on the output layer.\
For the compilation, the optimizer is `adam` and the loss function is `binary_crossentropy`.
- The model accuracy is under 75%. This is not a satisfying performance to help predict the outcome of the charity donations.

![d2-neuronscount.PNG](https://github.com/Praveeja-Sasidharan-Suni/Neural_Network_Charity_Analysis-/blob/main/images/d2-neuronscount.PNG?raw=true)

![d2-2output.PNG](https://github.com/Praveeja-Sasidharan-Suni/Neural_Network_Charity_Analysis-/blob/main/images/d2-2output.PNG?raw=true)
- The results are saved to an HDF5 file.

### Deliverable 3:
### Optimize the Model using the knowledge of TensorFlow
- To increase the performance of the model, we applied bucketing to the feature `ASK_AMT` and organized the different values by intervals.\
- Increased the number of neurons on the hidden layers

![d2-neurons.PNG](https://github.com/Praveeja-Sasidharan-Suni/Neural_Network_Charity_Analysis-/blob/main/images/d2-neurons.PNG?raw=true)
![d3-1result.PNG](https://github.com/Praveeja-Sasidharan-Suni/Neural_Network_Charity_Analysis-/blob/main/images/d3-1result.PNG?raw=true)

- Used a model with three hidden layers.\

![d2-2code.PNG](https://github.com/Praveeja-Sasidharan-Suni/Neural_Network_Charity_Analysis-/blob/main/images/d2-2code.PNG?raw=true)
![d3-2%20output.PNG](https://github.com/Praveeja-Sasidharan-Suni/Neural_Network_Charity_Analysis-/blob/main/images/d3-2%20output.PNG?raw=true)

- We also tried a different activation function (`tanh`) .
- Also tried increasing the number of epochs to the training regimen.

![d3-3code.PNG](https://github.com/Praveeja-Sasidharan-Suni/Neural_Network_Charity_Analysis-/blob/main/images/d3-3code.PNG?raw=true)
![d3-3output.PNG](https://github.com/Praveeja-Sasidharan-Suni/Neural_Network_Charity_Analysis-/blob/main/images/d3-3output.PNG?raw=true)

- The results are saved to an HDF5 file.
- 
### Deliverable 4:
- A Written Report on the Neural Network Model is submitted.

## Summary
The deep learning neural network model could reach upto 74.75% accuracy. Considering that this target level is pretty average we could say that the model is not outperforming.\
Since we are in a binary classification situation, we could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees 
to generate a classified output and evaluate its performance against our deep learning model.
