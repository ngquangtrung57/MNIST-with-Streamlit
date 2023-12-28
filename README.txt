# MNIST Digit Recognition

This repository contains code for training a convolutional neural network (CNN) to detect MNIST digits and a Streamlit web app for digit recognition.

## Files

- **`training.ipynb`**: Jupyter Notebook for the CNN model training.
- **`MNIST Dataset JPG format`**: Jupyter Notebook exploring OpenCV and reading .jpg MNIST files.
- **`streamlit_app.py`**: Python script for the Streamlit web app.
- **`my_model.pth`**: Saved model weights file.

## Model Training 

### Overview

The model is a simple CNN designed for MNIST digit recognition. It consists of two convolutional layers, followed by max-pooling and fully connected layers.

### How to Run
1. Install the required libraries
2. Open `training(1).ipynb` in Jupyter Notebook or Google Colab.
3. Run the cells to train the model.
4. The trained model will be saved as `my_model.pth`.

## Streamlit Web App

### Overview

The Streamlit web app (`streamlit_app.py`) allows users to draw a digit on a canvas, and the trained model will predict the digit.

### How to Run

1. Install the required libraries
2. Open a terminal and navigate to the project directory.
3. Run the Streamlit app: `streamlit run streamlit_app.py`.
4. Draw a digit on the canvas, and the app will predict the digit.

### Web App Features

- Users can draw digits on the canvas.
- The app displays the model input and predicts the drawn digit.
- Predictions are accompanied by a confidence level.
- The drawing history is shown in the sidebar.
- Users can clear the canvas and drawing history.


### Feel free to explore and customize the code to enhance the functionality of the model or web app.
