# **Behavioral Cloning** 

## Project Description
The goal of this project is to use a deep learning model to clone the driving behavior of a driver/operator for the car in a simulated environment. This end-to-end learning allows the car to drive around the track autonomously without leaving the road. The model is based on this work by [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and have been found to work for this domain.


---


The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

---
### Structure of repository

* [model.py](https://github.com/niteshjha08/Behavioral-Cloning/blob/main/model.py) containing the script to create and train the model
* [drive.py](https://github.com/niteshjha08/Behavioral-Cloning/blob/main/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/niteshjha08/Behavioral-Cloning/blob/main/model.h5) containing a trained convolution neural network 
* [images](https://github.com/niteshjha08/Behavioral-Cloning/tree/main/images) contains images of simulations and testing.

### Data Collection and Training Strategy
The training of the model is based on images collected by manually driving multiple laps around the track. For this, Udacity's Unity simulator is used which can be found [here](https://github.com/udacity/self-driving-car-sim). Images from the car's center, left, and right cameras are captured along with the corresponding steering angle and throttle. 
To capture ideal driving behavior, center lane driving was done for two laps. This is an example of center lane driving. 
<p align="center">
  <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/center_lane.PNG" height="280"/>
</p>

In addition to ideal center lane driving, the model should also be able to recover in case it deviates from the lane center, and continue on the road instead of slowly drifting off. For this, images are also recorded for the recovery from left and right sides of the lane towards center. The images of recovery are shown below:

<div class="row">
  <div class="column2">
    <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/left_recovery.PNG" alt="Left_recovery" style="width:100%">
  </div>
  <div class="column2">
    <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/right_recovery.PNG" alt="Right_recovery" style="width:100%">
  </div>
</div>

<style type="text/css" rel="stylesheet">
column3 {
  float: left;
  width: 33.33%;
  padding: 5px;
}
column2 {
  float: left;
  width: 50%;
  padding: 5px;
}
row::after {
  content: "";
  clear: both;
  display: table;
</style>

This is done for both tracks to collect more data points. To generalise the model, collected images are augmented by flipping them horizontally and reversing the sign of steering angle( +ve --> -ve, -ve --> +ve).

<div class="row">
  <div class="column2">
    <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/center.PNG" alt="Straight" style="width:100%">
  </div>
  <div class="column2">
    <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/flipped.PNG" alt="Flipped" style="width:100%">
  </div>
</div>

We are capturing images from the left and right cameras as well. These images can be used to augment our data points and capture the scene with an off-center shift. These additional images will also help the model develop ability to recover towards the center of the road. The following are images showing images at a particular instance from three diffent cameras.

<div class="row">
  <div class="column3">
    <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/left.PNG" alt="Left" style="width:100%">
  </div>
  <div class="column3">
    <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/center.PNG" alt="Center" style="width:100%">
  </div>
  <div class="column3">
    <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/right.PNG" alt="Right" style="width:100%">
  </div>
</div>

However, the trained model will take the center camera's image as the input and produce the output steering angle.  Thus, we require 3D scene transformation information to calculate the optimal correction factor for steering commands for these off-center images. For a car driving straight in the center of the road, the left camera will see the scene as though it was closer to the left edge and correspond this image to a steering angle of zero! At the same time, the right image will see the car closer to the right and associate it to 0 steering angle as well. This will cause abrupt behavior in the model. This problem is illustrated below for a left turn.
<p align="center">
  <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/multi_camera_problem.PNG" height="280"/>
</p>

Thus, we approximate a correction factor (by trial and error) to 0.3. We add this factor to the left images so that when testing our model on center camera images, it will associate a slight right turn to images closer to the left. Similarly, we subtract this factor from the right images. This technique was employed in [this paper by NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).



### Model Architecture

The model is based on this NVIDIA end-to-end learning model for this task. It consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 64. The complete architecture is defined below:

| Layer        | Output shape      | No. of Parameters |
|:------------:|:-----------------:|:----------:|
| Convolution  | (None,76,316,24)  | 1824     | 
| Maxpooling   | (None,38,158,24)  | 0        |
| Convolution  | (None,34,154,36)  | 21636    |
| Convolution  | (None,30,150,48)  | 43646    |
| Convolution  | (None,28,148,64)  | 27712    |
| Convolution  | (None,26,146,64)  | 36928    | 
| Flatten      | (None, 243944)    | 0        |
| Dense        | (None, 500)       | 121472500|
| Dropout      | (None,500)        | 0        |
| Dense        | (None,100)        | 50100    |
| Dense        | (None,50)         | 5050     |
| Dense        | (None,10)         | 510      |
| Dense        | (None,1)          | 11       |



The model contains dropout layers in order to reduce overfitting, which drops out 50% of activations. 

The model was trained on a total of 24108 images and validated on 20% of the dataset to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model was tuned and the final parameters were:
* No. of epochs = 5
* Optimizer = Adam
* Steering correction factor = 0.3
* Batch size = 32
* Learning rate = 0.001(default)
* Validation split = 0.2

### Preprocessing
The images captured were first converted from BGR to RGB since cv2 library reads them in BGR format and drive.py uses PIL library which loads them in RGB.
The training images were cropped to remove 60 pixels from the top which included the sky, trees and mountains, and 20 pixels from the bottom which showed the car's bonnet/hood. Here is an original and cropped image.

<div class="row">
  <div class="column2">
    <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/uncropped.PNG" alt="Uncropped" style="width:100%">
  </div>
  <div class="column2">
    <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/cropped.PNG" alt="Cropped" style="width:100%">
  </div>
</div>
The images were then normalized to improve the speed of convergence.

### Training process and Testing
To feed the training data to the model, generators were used to avoid in-memory usage of the the huge dataset. Batches of 32 images were fed and the model was trained, the resulting model was saved in `model.h5`.

To test the model, the simulator was opened in the 'autonomous mode', which basically takes in input from `drive.py`. This file uses `model.h5` to predict a steering angle from the input camera image. 

On optimal training and model architecture, the car was able to stay on the road indefinitely. 
Note: Images can be captured while testing the model by specifying the command-line argument for directory of storage while running drive.py. The file `video.py` can be used to generate a video from these images.
Here is a short clip of the autonomous run.
<p align="center">
  <img src="https://github.com/niteshjha08/Behavioral-Cloning/blob/main/images/onboard_cam.gif" />
</p>
For the full video, head over to this [YouTube video!](https://youtu.be/SrQNSat_4bE)

### Conclusion
This is a simple example of using end-to-end learning to enable the car to steer itself by learning from driving behavior. It gives an idea of the range of tasks that can be accomplished by such a model, such as mimicking the driving smarts in this case.
 

