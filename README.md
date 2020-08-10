# **Behavioral Cloning** 


The goal this project is to use a convolution neural network in Keras to clone my own driving behavior in a simulated environment.

You can find the test video [here](https://youtu.be/Uq3W2EAp3pc) 


---

## Model Architecture and Training Strategy

### An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64. The original model architecture is designed by Nvidia, which can be found [here](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)

The model includes ELU layers to introduce nonlinearity while reduce the effect of dead nurons. 

The data is normalized in the model using a Keras lambda layer with equaiton:

pixel = (pixcel/ 127.5) - 1. 

### Attempts to reduce overfitting in the model

The model also contains dropout layers in order to reduce overfitting.

Before the training start, the sample images are seperated into training and validation datasets with a ratio of 0.2. Afterwards, both datasets are passed into a generator for batched sample loading to reduce RAM accessing.

### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and data augmentation method: mirroring images and negate corresponding labels.

## Model Architecture and Training Strategy

### Solution Design Approach

My first step was using the famous convolution neural network model: LeNet-5 with the trained parameters because I thought it already contained low layer detections like edges or curves that I didn;t have to train from scratch.

In order to evaluate how was the model , I split my image from center camera and steering angle data into a training and validation set. I found that my first model had both low mean squared error on the training set and the validation set. This implied that the model was underfitting. 

Therefore, I collect more data and observed that the loss from training and validation set drops smoothly. However, the model keep on failing to redurect with the correct angle on big turns, even if I feed more data on corner turning. Then I realize this might resulted from the insufficient parameters in my model. 

Then I changed my model to Nvidia's approach: 5 convolutional layer followed by 4 fully coneected layer.

Training the new model again, the phenomenon of overfitting occured, which is low training loss and high validation loss, and my car keeps on shaking even the road is straight.  
The final step I took was to add a dropout layer after the last convolution layer. The immediately benifit was remarkable! The car drove smoothly even during turning.

At the end of the process, the vehicle is able to drive autonomously around the track for no matter how many times without leaving the road.

### Creation of the Training Set & Training Process

To capture my own driving behavior, I first played with the simulator to make sure I myself is familiar with the system and can be a good driver. Then I started developement cycle:

1. Record for a few laps
2. load the previous model wieghts and train
3. test run 

During the test run, I found that the car speed in autonomous mode is around 9 miles/hr and in order to make the steering factor aligned well, I recollected my data with speed limited to around 9 miles/her as well. In particular, the new collected data contained 2 laps in the middle of the road, 1 lap that contained fixing from wrong direction, and 1 reversed lap.

Also, I doubled my input images with mirroring all the images and negating their labels so that the input will contain equal number of left and right turns. In addition, I cropped the image to retain only the road part to save more memory and remove uncessary information.

Here is a pair of mirrored image:

![](https://i.imgur.com/zWrGQA9.jpg)
![](https://i.imgur.com/sFCAvWy.jpg)

And here is the cropping result:

![](https://i.imgur.com/1j2KTFi.jpg)
![](https://i.imgur.com/lwLI7JI.jpg)


After shuffling and seperating dataset into training and validation, The car can drive itself endlessly whithout running out of the road!

