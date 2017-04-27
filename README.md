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

My model consists of 5 CNNs and 3 fully connected NNs. (model.py line 106 ~ 120)
Each stage includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (model.py line 107). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 117). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 28). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with the learning rate 0.0001 (model.py line 16).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used default date set provided by udacity using image flipping for recovering from the left and right sides of the road.


### Architecture and Training Documentation

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use CNN for recognizing the track and then NN for prediction of steering angle.

My first step was to use Nvidia model. I thought this model might be appropriate because they tried to solve the same problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I plot the training loss and validation loss to check overfitting (model.py lines 157 ~ 168).

To combat the overfitting, I modified the model so that the fully connected NN had dropout with 0.5 of keep_prob (model.py line 117). I also added keras callback for early stopping (model.py line 144)

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added side camera images and then also added flipped images of them (model.py lines 65 ~ 93). When I adjusted angles of the side camera images by adding 0.5, it worked good.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

The final model architecture (model.py lines 104-120) consisted of 5 CNN layers and 3 NN layers as the following.

| Layer         |     Description	      | 
|:-------------:|:-----------------------:| 
|Input          | 90 x 320 x 3 RGB image  |
|Normalization  |                         |
|ConvNet        | 5x5@24                  |
|RELU           |                         |
|Max pooling    | 2x2                     |
|ConvNet        | 5x5@36                  |
|RELU           |                         |
|Max pooling    | 2x2                     |
|ConvNet        | 5x5@48                  |
|RELU           |                         |
|Max pooling    | 2x2                     |
|ConvNet        | 3x3@64                  |
|RELU           |                         |
|ConvNet        | 3x3@64                  |
|RELU           |                         |
|Fully Connected| 100                     |
|RELU           |                         |
|Dropout        | 0.5                     |
|Fully Connected| 50                      |
|RELU           |                         |
|Fully Connected| 10                      |
|RELU           |                         |
|Output         | 1                       |

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one. But it was a little hard to control. So I decided to just use the data provided by udacity.

I randomly shuffled the data set and put 20% of the data into a validation set.

After data splitting, I augmented the training data. I flipped images and angles and addded the left and right sided camera images for recovery. 

After that process, I had 38568 number of training data points. I then preprocessed this data by cropping and normalization (model.py lines 106 ~ 107).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25. I used an adam optimizer with 0.0001 of learning rate parameter for fine tuning and for early stopping, I used keras callback with 3 of patience parameter (model.py line 144).
