# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
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
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

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
* The number of fully connected layers been reduced to reduce model size.

The

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

I found this to be important because comparing with large number of parameters, the amount of training set is still limited.

The model was trained and validated on different data sets to ensure that the model was not overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

I used both three cameras for the training. In addition, I flipped all images horizontally to get twice the data.

I collect training data from both tracks and both directions (clockwise and counter-clockwise).


### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a linear model. This is a simple model and I did not it expect it to work. But I choose this as a start point to see how a simple network fail.

The result of a linear model is the car will simple make a U-turn with small radius. This can be understood as the training data is a car always running within a large circle and keeps turning left. So my car, after averaging all images, just keeps turning left.

Then I used a covn neural network similar to Nvidia model, which is working according to my classmates.

The final step was to run the simulator to see how well the car was driving around track one and two. The first run worked quite well, my car keep in the track for a long time.

Then I collected more data on track one and track two. And I did a few changes to my model (add dropout and reduce fully connected layer, etc).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

This is the fist submitted model architecture I used for this project.

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


To achieve a better results, I modified my model based on discussion with other on forum. The final model architecture is the following:
(see model.py line 91 to 111)

* Crop
* Conv2D, 3x1x1, with subsample 1x1
* Conv2D, 24x5x5, with subsample 2x2
* Conv2D, 36x5x5, with subsample 2x2
* Dropout(0.2)
* Conv2D, 48x5x5, with subsample 2x2
* Dropout(0.2)
* Conv2D, 64x3x3, with subsample 2x2
* Conv2D, 64x3x3, with subsample 2x2
* Dropout(0.2)
* Fully connected layer 1164
* Fully connected layer 100
* Fully connected layer 50
* Fully connected layer 10
* Fully connected layer 1


At first, I add tanh as the activation function in the final fc layer. The reason is the final output, which is normalized steer angle, is ranged from -1 to 1. Thus, by using a tanh function at the end, I can limited the final output.

However, after a few more trying, I found that when using a tanh activation function, the model will less likely to make a large turn, which in some conditions will cause the car driving outside the lane. Thus, in the finaly model I remove this activation.

In addition to that, I also added l2 regularizers to all layers, to further reduce overfitting. Considering I am now using the data collected from track 1, this is even more important.

I tested different values for dropout, including 0.1, 0.2 and 0.5. Then I chose the best performance one, 0.2.


#### 3. Creation of the Training Set & Training Process

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

**Note: There are some modifications to my training and testing after first submitting.**


1. I remove data collected on track 2 to get a better performance.

2. I found training data collected with different simulator graphic setting will strongly effect the training results. So, I removed all my previous training data, which were collected with "fastest" setting, and replaced them with newly collected data on track 1 with better graphic setting in the simulator.

3. I modified the drive.py to accept openCV formatted images, to match the training process.


#### 4. results

I tested my car on both tracks, as shown in \run1,  \run2 and \run3

On track one, car can stay for a long long long time without leaving track, which can be seen in video file  in folder \run1 and \run3.

On track two, currently car can only stay for a short time (a few minutes), and then out of road. This is because data collected on track two is quite limited comparing with track one.
