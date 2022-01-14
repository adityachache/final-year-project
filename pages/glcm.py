import streamlit as st
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.feature import greycomatrix, greycoprops

def app():
    
    st.write("""
        __Texture extraction with input image__
    """)
    ########## Load image ##########
    def load_image(img):
        out_img = Image.open(img)
        return out_img

    def showImage(img):
        plt.figure(figsize=(8,8))
        plt.imshow(img,cmap=None)
        plt.xticks([]),plt.yticks([])
        plt.show()

    # Image input
    uploaded_image = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])
    if uploaded_image:
        img_input = load_image(uploaded_image)
        # save image color
        with open(os.path.join("./pages/temp-images", uploaded_image.name), "wb") as f:
            f.write(uploaded_image.getbuffer())
            st.caption("Image saved")

        color_image_input = "./pages/temp-images/"+uploaded_image.name
        img_input_gray = cv2.imread(color_image_input, cv2.IMREAD_GRAYSCALE)
        col1, col2 = st.columns(2)
        with col1:
            st.image(color_image_input, caption="Original image")
        with col2:
            st.image(img_input_gray, caption="Gray image")
        
        with st.expander("View co-occurence matrix"):
            glcm = greycomatrix(img_input_gray, [2, 8, 16], [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6], 256, symmetric=True, normed=True)
            st.write(glcm)

        