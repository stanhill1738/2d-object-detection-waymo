# 2D Object Detection using the Waymo Open Perception Dataset


## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
This project focuses on building a machine learning model that can detect and classify objects (vehicles, pedestrians, and cyclists) in camera images from self-driving cars. It uses real-world driving footage from the Waymo Open Dataset to train a computer to draw boxes around these objects and identify what they are. The goal is to help improve the perception systems used in autonomous driving technology.

## DATA
The project uses the **Waymo Open Dataset (Perception)**, which includes millions of frames from autonomous vehicle sensors.
The labels focus on three object classes: **vehicles**, **pedestrians**, and **cyclists**. Data was pre-processed into PyTorch `.pt` files, and subsets were sampled for training, validation, and testing.  
**Citation:** [Waymo Open Dataset](https://waymo.com/open/)

## MODEL 
The model architecture is **Faster R-CNN** from the PyTorch `torchvision` library, selected for its strong performance on object detection benchmarks and out-of-the-box support for fine-tuning. It supports multiple object classes and bounding box regression, making it ideal for 2D perception tasks.

## HYPERPARAMETER OPTIMSATION
Hyperparameters such as learning rate, momentum, weight decay, and batch size were optimised using **Optuna**, an automated hyperparameter search framework. The objective function was based on **mAP@0.5** on a validation set.  
Experiments were run on a GCP L4 GPU VM, with training data subsets sampled locally to improve speed and stability.

## RESULTS
A summary of your results and what you can learn from your model 

You can include images of plots using the code below:
![Screenshot](image.png)