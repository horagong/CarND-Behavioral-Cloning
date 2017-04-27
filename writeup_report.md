**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 CNNs and 3 fully connected NNs. (code line 106 ~ 120)
Each stage includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 107). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 117). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 28). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with the learning rate 0.0001 (model.py line 16).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used default date set provided by udacity using image flipping for recovering from the left and right sides of the road.


### Solution Design Approach

I decided to use Nvidia model because it has the same purpose.

What was important was how to choose the training data. I saw that some people say that it is possible to finish the project with only the data provided by udacity, even on the second challenge track. I wanted to do that way.

That was not that easy. I first used flipping the image for data augmentation and adjusting angle into the left and right sided camera images for recovery. As tuning the parameters, it was possible to run on the first track.

But it failed on the steep second track. It will be possible to run the second track doing the same method with training data of the second track. But I wanted to make it with only the first training data, so kept trying after due date. I tried to augment the turning angles so to get adaptive to the steep curve. It made better but not enough.

After trying some methods, I got to doubt that it is really possible to pass the second track with only the first track training data. It might be possible but, that way is, I think, a bit cheating and cannot be generalized. I think end-to-end approach in drving is not practical in that the traning data would have restricted coverage in the real world. I believe this end-to-end approach is for a complement when the main methods fails in unexpected situations. So I submit it without passing the second track.