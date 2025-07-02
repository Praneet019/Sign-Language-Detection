# Sign-Language-Detection
This project trains a convolutional neural network model to recognize American Sign Language (ASL) alphabets and a few custom signs like Hello and Yes using a custom dataset collected with a webcam. The model supports real-time prediction using webcam and image upload through a graphical user interface built with Tkinter.

## Project Contents

train_sign_model.py script for training the model on the custom dataset

sign_gui.py script for running the GUI for real-time webcam or image upload prediction

model folder containing the trained model file sign_model.h5 and the label_encoder.pkl file

requirements.txt file listing required Python packages

## Dataset

This is the custom dataset I created and used for this model. Ensure you make changes in your code with the path you save the dataset and other folders.

You can access the dataset here: 

## Features

Real-time sign prediction with webcam

Image upload support for predicting signs on images

Hand detection and tracking with MediaPipe

Time restriction feature to allow application access only between 6 PM and 10 PM

## How to Train
Run train_sign_model.py in the project folder after organizing your custom dataset into labeled subfolders for each sign inside a folder named custom_dataset. The trained model and label encoder will be saved in the model folder.

How to Run the GUI
Run sign_gui.py in the project folder to open the graphical interface. Choose either Upload Image or Start Webcam. Press q in the webcam window to quit.

Requirements
Install dependencies using pip install -r requirements.txt

Notes
For best results, train the model with a custom dataset containing clear and consistent images of each sign. Make sure the label encoder does not contain unwanted labels like unused classes. The custom dataset folder should not be uploaded to GitHub if it is large.
