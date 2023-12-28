import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from streamlit_drawable_canvas import st_canvas

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
model.load_state_dict(torch.load('replace with your model path',map_location=torch.device('cpu')))
model.eval()

st.title('My MNIST model')
st.markdown('''
Try to write a digit!
''')


SIZE = 192

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')

drawing_history = []

if canvas_result.image_data.any():
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    st.write('Model Input')
    st.image(rescaled)


if st.button('Predict'):
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bwx = torch.from_numpy(test_x.reshape(1, 28, 28)/255)
        val = model(bwx.float().unsqueeze(0))
        prediction = np.argmax(val.detach().numpy()[0])
        st.write(f'Result: {prediction}')

        confidence = np.exp(val.detach().numpy()[0][prediction])
        if confidence > 0.8:
            st.balloons()
            st.success("The model is confident in its prediction!")
        else:
            st.warning("The model is not very confident in its prediction.")

  
        st.bar_chart(np.exp(val.detach().numpy()[0]))

 
        if prediction == 0:
            st.image('https://media.giphy.com/media/3o6Zt481isNVuQI5tm/giphy.gif', use_container_width=True)
        elif prediction == 1:
            st.image('https://media.giphy.com/media/3oEduXklpTRxPxFyZ2/giphy.gif', use_container_width=True)
        drawing_history.append((rescaled, prediction))
        
        drawing_history = drawing_history[-5:]
st.sidebar.header('Drawing History')
for idx, drawing_entry in enumerate(drawing_history):
    drawing_img, prediction = drawing_entry
    st.sidebar.image(drawing_img, caption=f'Drawing {idx+1}\nPrediction: {prediction}', use_column_width=True)


if st.sidebar.button('Clear History'):
    drawing_history = []
    st.sidebar.success('History cleared!')

if st.sidebar.button('Clear Canvas'):
    drawing_history = []
    st.sidebar.success('Canvas cleared!')

