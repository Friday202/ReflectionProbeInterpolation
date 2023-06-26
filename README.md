# Reflection probe interpolation with neural network

This project aims to (fix) generate new images using an encoder/decoder neural network trained on Unreal Engine-generated images. Initially, the idea was to utilize Unreal Engine to capture images and train the neural network based on the distances from reference points. The captured image would undergo bi-linear interpolation using current actor location (and its triangle), and then be fed into the neural network. The network was specifically trained to refine the interpolated image by aligning it with the provided ground truths.

Due to difficulties in accessing cubemaps in Unreal Engine the trained model was tested in Python instead. For more detailed information on the project, please refer to the provided report.

# Installation
The project is best cloned or downloaded as ZIP as for any modification you will need to change source code. 

# Features
- Unreal engine script written in Python for Unreal console to extract images from level.
- Model training and architecture using Keras Tensorflow (TF with CUDA enabled is recommended).
- Preprocess data script for processing extracting images.
- Visualizing script to evaluate model.


# Results

Bi-linear interpolation - input           |   Generated ouput 
:-------------------------:|:-------------------------:
![](https://github.com/Friday202/ReflectionProbeInterpolation/blob/main/Results/animationBI4.gif)  |  ![](https://github.com/Friday202/ReflectionProbeInterpolation/blob/main/Results/animation.gif)


Although the model was not tested in Unreal Engine the application of the project scopes beyond just Unreal Engine. If given enough captured data we can train a neural network to generate additional images. This could be potentially used in other domains such as geo-captured images, (e.g. Google street view). 

## Note
Please note that the dataset is not included in the repository as it is too large. 
