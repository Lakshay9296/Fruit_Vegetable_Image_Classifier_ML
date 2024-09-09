import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
st.set_page_config(page_title="Image Classifier", page_icon=":Search:")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Fruit_and_Vegetable_Identifier.keras')
    img_width = 180
    img_height = 180
    return model, img_width, img_height

model, img_width, img_height = load_model()

categories = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
              'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
              'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'pear',
              'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'spinach', 'sweetcorn',
              'sweetpotato', 'tomato', 'turnip', 'watermelon']

st.sidebar.title("About This Project")
st.sidebar.markdown("""
This project is an image classifier designed to identify various fruits and vegetables from images.

**Technologies Used:**
- **TensorFlow**: For building and training the deep learning model.
- **Keras**: To define the architecture of the Convolutional Neural Network (CNN).
- **NumPy**: For numerical operations.
- **PIL**: For image processing.
- **Streamlit**: For creating an interactive web application.
- **Matplotlib**: For plotting (if needed).

For more details on the implementation, check out the project repository on [GitHub](https://github.com/Lakshay9296/Fruit_Vegetable_Image_Classifier_ML).
""")

st.sidebar.header("Contact")
st.sidebar.markdown("""
For any inquiries or feedback, feel free to reach out:

- **Email**: lakshay.kumar9911@gmail.com
- **LinkedIn**: [Lakshay Kumar](https://www.linkedin.com/in/lakshay9911)
""")

st.title("Image Classifier: Fruits/Vegetables", anchor=False)

uploaded_file = st.file_uploader("Choose an Image", type=('.jpg', '.jpeg', '.png'))

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)
    image_resized = image.resize((img_width, img_height))
    img_arr = tf.keras.utils.img_to_array(image_resized)
    img_batch = tf.expand_dims(img_arr, 0)
    predict = model.predict(img_batch)
    score = tf.nn.softmax(predict)
    st.image(image, caption='Uploaded Image.')
    st.subheader(f'Vegetable/Fruit: {categories[np.argmax(score)].upper()}', anchor=False)
    st.subheader(f"Accuracy: {np.max(score) * 100:.2f}%", anchor=False)
