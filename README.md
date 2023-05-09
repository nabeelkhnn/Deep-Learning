# Deep-Learning
IDS-576, University of Illinois Chicago

Overview of Python Projects

This repository contains the following Python projects that I completed as part of my coursework or personal projects:

#### Assignment 1: Neural Network Implementation
The assignment consists of several parts related to machine learning and neural networks. Here is a brief overview of each task:

Backpropagation: Drawing a computation graph for a given function and computing its gradient with respect to its inputs.
Gradient Descent: Implementing a linear regression model with one and two features and using backpropagation and gradient descent to find the model parameters. The mean squared error is used as the loss function. The performance of the model is evaluated by plotting the error as a function of the number of iterations and comparing it with the true function.
ML Basics: Implementing a function to compute the multiclass logistic loss with L1 and L2 regularization.
Classification Pipeline: Implementing a linear classifier using multiclass logistic loss with L2 regularization and evaluating its performance on a given dataset. The sensitivity of the model's performance to different learning rates and regularization parameter values is evaluated using suitable plots.
Feedforward Neural Networks: Implementing two models of feedforward neural networks with different activation functions and using them to solve a classification problem for MNIST/Fashion MNIST. The influence of optimizer choice and the number of hidden units on the performance is discussed.

#### Assignment 2: Convolutional Neural Networks
1. CNN's and finetuning
This project involves using convolutional neural networks (CNNs) and finetuning to perform image classification on the CIFAR 10 dataset. Specifically, the project uses the pretrained Resnet18 model to extract features from the images, which are then used as inputs in a new multi-class logistic regression model. The performance of this model is evaluated and compared to the performance of a finetuned Resnet18 model. The project includes a description of any choices made during the process and displays the top 5 correct and top 5 incorrect predictions for each class of images.

2. Movie embeddings
For this project, we will use the MovieLens small dataset to create embeddings for movies, which will allow us to recommend similar movies based on user ratings. We will use the number of users that liked both movies as the data Xij, and optimize a cost function to obtain the movie embeddings.

To optimize the cost function, we will use gradient descent with various learning rates and optimizers, and plot the loss as a function of iteration. After obtaining the movie embeddings, we will recommend the top 10 movies given a set of input movies, such as Apollo 13, Toy Story, and Home Alone. We will describe our recommendation strategy and investigate whether the recommendations change with different learning rates and optimizers.

#### Assignment 3: Transfer Learning
The project involves implementing two different natural language processing tasks using recurrent neural networks (RNNs). The first task involves building two language models using the torchtext IMDB dataset - one based on Markov (n-gram) model and the other based on Long Short-Term Memory (LSTM) model. The training performance of both models will be plotted as a function of epochs/iterations. The key design choices for each model will be described, and how each choice influences training time and generative quality will be discussed. The generated reviews will be evaluated by starting with the phrase "My favorite movie" and generating an approx. 20 word review five times for each model.

The second task involves training two sequence to sequence models - one for a chosen language to English translation and the other for the reverse translation. The GloVe 100 dimensional embeddings will be used in the reverse translation model. Five well-formed sentences from the English vocabulary will be input to the reverse translation model, and the resultant translated sentences will be input to the chosen language to English translation model. All model outputs in each case will be displayed.

#### Major Project: Real-time Object Detection using YOLOv5
In this project, I implemented real-time small object detection using YOLOv5. I trained the model on a custom dataset of small objects and tested it on airport images. The notebook includes code for preprocessing the data, defining the model architecture, and evaluating its performance. I also included code for visualizing the detections.
