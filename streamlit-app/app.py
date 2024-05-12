import cv2
import easyocr
import streamlit as st
import helper
import pickle
from PIL import Image
import os

# Load the ML model
model = pickle.load(open('model.pkl', 'rb'))

# Function to process image input using EasyOCR
def process_image_input(uploaded_file):
    # Save the uploaded file temporarily
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Read the temporary image using OpenCV
    img = cv2.imread(temp_image_path)

    # Delete the temporary image file
    os.remove(temp_image_path)

    # Process the image with EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    text_ = reader.readtext(img)
    detected_text = [t[1] for t in text_]
    return detected_text
  

# Create a Streamlit app
st.set_page_config(layout="wide")
st.title('Question Duplicate Detection')

st.sidebar.title('Options')
option = st.sidebar.selectbox('Choose Input Source', ('Text Input', 'Image Input'))

if option == 'Text Input':
    q1 = st.text_input('Enter question 1')
    q2 = st.text_input('Enter question 2')

    if st.button('Find'):
        query = helper.query_point_creator(q1,q2)
        if(q1 == '' or q2 == ''):
            st.warning('Please enter both questions')
            # st.stop()
        else:
            result = model.predict(query)[0]
            st.success('Prediction Successful')
            if result:
                st.markdown('#### Duplicate')
            else:
                st.markdown('#### Not Duplicate')

elif option == 'Image Input':
    st.header('Upload Image 1')
    uploaded_file_1 = st.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"])

    st.header('Upload Image 2')
    uploaded_file_2 = st.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"])

    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        st.image(uploaded_file_1, caption='Uploaded Image 1', use_column_width=True)
        st.image(uploaded_file_2, caption='Uploaded Image 2', use_column_width=True)
        
        if st.button('Process Images'):
            detected_text_1 = process_image_input(uploaded_file_1)
            detected_text_2 = process_image_input(uploaded_file_2)

            #  st.write("Detected Text from Image 1:")
            # for text in detected_text_1:
            #     st.write(text)
                
            # st.write("Detected Text from Image 2:")
            # for text in detected_text_2:
            #     st.write(text)

            query = helper.query_point_creator(detected_text_1, detected_text_2)
            result = model.predict(query)[0]

            if result:
                st.header('Duplicate')
            else:
                st.header('Not Duplicate')
            
