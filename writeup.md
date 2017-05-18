#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[training1]: ./writeupimg/center_2017_05_16_18_44_09_891.jpg "t1"
[training2]: ./writeupimg/center_2017_05_17_15_15_44_529.jpg "t2"



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model I used is based on Nvidia Autopilot model. It consists of the following layers:


* Conv2D, 36x5x5, with subsample 2x2
* Conv2D, 48x5x5, with subsample 2x2
* Conv2D, 64x5x5, with subsample 1x1
* Conv2D, 64x3x3, with subsample 1x1
* Fully connected layer 400
* Fully connected layer 100
* Fully connected layer 50
* Fully connected layer 10
* Fully connected layer 1

After each Conv2D, I added dropout layer with dropout prob 0.2 to reduce overfitting.

Comparing with original Nvidia model, the main changes are:

* Dropout layer been added
* The number of fully connected layers been reduced.

The

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

I found this to be important because comparing with large number of parameters, the amount of training set is still limited.


The model was trained and validated on different data sets to ensure that the model was not overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

I used both three cameras for the training. In addition, I flipped all images horizontally to get twice the data.


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a linear model. This is a simple model and I did not it expect it to work. But I choose this as a start point to see how a simple network fail.

The result of a linear model is the car will simple make a U-turn with small radius. This can be understood as the training data is a car always running within a large


to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

This is the final model architecture I used for this project.

* Crop
* Conv2D, 36x5x5, with subsample 2x2
* Dropout(0.2)
* Conv2D, 48x5x5, with subsample 2x2
* Dropout(0.2)
* Conv2D, 64x5x5, with subsample 1x1
* Dropout(0.2)
* Conv2D, 64x3x3, with subsample 1x1
* Dropout(0.2)
* Fully connected layer 400
* Fully connected layer 100
* Fully connected layer 50
* Fully connected layer 10
* Fully connected layer 1

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used both track one and track two for centered
lane driving.

Here is a example of an image captured on track one. This image is taken when I drive the car counter-clockwise.

![alt text][training1]

Here is a example of an image captured on track two.

![alt text][training2]


In addition, I trained the simulator to capture how to drive back to center by recording off-centered driving.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles. This is proven to be working.

In order to get more off-center-driving data, I used additional cameras from left and right to get more off-centered data.

After the collection process, I had roughly 9000 number of data points. Then I normalized all input images.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I cannot load all data into the memory of my PC, so I defined a generator that load a batch of images and process them (flipping, etc)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I chose the number of epoch to be 5 because after 5 validation loss start to increase, which indicates overfitting.

### results

I tested my car on both tracks, as shown in \run1 and \run2.

On track one, cars can stay for a long long long time without 
