
# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Code and Usage

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results**

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

---




## Steps
**Here I describe  my implementation of the project. **

### Model Architecture and Training Strategy

#### 1. Model architecture design

My model consists of 5 a convolution neural network layers (model.py lines 24-48)，and 5 fully connnected layers(model.py lines 53-66). the data is normalized in the model using a Keras lambda layer (code line 22), and Max pooling is add after each convolution layers. To introduce nonlinearity , the model use RELU activation.  


#### 2. Methods to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 73-74). and each epoch the training data num is 20032(75.8%) and the validation data num is 6400(24.2%). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, the learning rate was set 0.0001 (model.py line 15).

#### 4. Training data generation

Training data was chosen to keep the vehicle driving on the road. I used a combination of clockwise lane driving and counter-clockwise driving, some random data generations were added to get more data

For details about how I created the training data, see the next section. 



### Model Architecture and Training Strategy

#### 1. Solution Design Approach
First I have tried LeNet since it is a good start. The car is able to make the first turn but it is get off befor the first bridge, I modified LeNet into deeper network and add two more 5×5×64 convolution layers, and it can get through the bridge, but went off the road after the bridge. The LeNet is too shallow and its repretation abilities is limited, so I try from  the NVIDIA model, which has been used by NVIDIA for the end-to-end self driving test. As such, it is well suited for the project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.To speed up the training process, I crop the image rows of the trees, sky and and water, modified the image input to 64×64×3, and normalize the image data. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I add random adversity to trainning data, randomly shear, flip, rotate and augament the image.At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

[//]: # (Image References)

[image1]: ./examples/conv_architecture.png "Model Visualization"


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-68) consisted of 5 a convolution neural network layers and 5 fully connected layers.Here is a visualization of the architecture 

![alt text][image1]

[image2]: ./examples/counter-clockwise.jpg "Counter-Clockwise Image"
[image3]: ./examples/clockwise.jpg "Clockwise Image"

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using counter-clockwise lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle two laps on track using clockwise lane driving as below:

![alt text][image3]

[image4]: ./examples/sheared.png "sheared Image"
[image5]: ./examples/cropped.png "Cropped Image"
[image6]: ./examples/flipped.png "Flipped Image"
[image7]: ./examples/resized.png "Resized Image"
[image8]: ./examples/jungle.jpg "Jungle Image"

To augment the data set, I also randomly sheared, randomly croped and randomly flipped images,So I can get endless data. For example, here is an image that has then been sheared:
![alt text][image4]

the sky and trees are not releate to the lane keeping, to speed up the training, we cropped the image. 35% of the original image from the top and 10% image from the bottom are removed. here is an image that has then been cropped:
![alt text][image5]

To make the model not bias to the driving orientation, we flipped the image and the measurement. there are 0.5 probability that a image is flipped. here is an image that has then been flipped:
![alt text][image6]


To speed up the training process, smaller data will be a good option, so image is resize into 64×64×3. here is an image that has then been resized:
![alt text][image7]

To make the model make self-driving in the jungle lane , I also get training data from the jungle lane and make same augmentation to it, Since I drive slowly in the jungle lane, it can only drive at 15km/h speed. here is an image from jungle:
![alt text][image8]

Finally, In each epcho, I get randomly 20032 images data and 6400 images data. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
Finally, we tried epochs  such as 5, 10, 25, 30 and 100. However, **30** works well on both training and validation tracks.
