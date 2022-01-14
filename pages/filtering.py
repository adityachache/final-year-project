import streamlit as st
import cv2
import numpy
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage

def app():
    st.write('# Image Filtering')
    st.write("""

    The goal is to apply and compare different image filtering algorithms, 
    `linear and non-linear`, on the input images shown below.

    """)
    img_or = ['images/295087.jpg', 'images/3096.jpg', 'images/42049.jpg', 'images/175032.jpg', 'images/189080.jpg']

    # load images function
    def load_image(img):
        out_img = Image.open(img)
        return out_img

    # filter input image
    def filter_input(img):
        img_rgb = cv2.imread(img)
        mean = cv2.blur(img_rgb, (3,3))
        median = cv2.medianBlur(img_rgb, 5)
        # sobel
        gray_img = cv2.cvtColor(img_rgb , cv2.COLOR_BGR2GRAY)
        gauss_blur_img = cv2.GaussianBlur(gray_img, (5,5), 0)
        x = cv2.Sobel(gauss_blur_img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(gauss_blur_img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("__Original__")
            st.image(img)
        with col2:
            st.write("__Mean image__")
            st.image(mean)
        with col3:
            st.write("__Median Image__")
            st.image(median)
        with col4:
            st.write("__Sobel image__")
            st.image(sobel)
    
    st.write("### Image filtering with user input")
    input_image = st.file_uploader("Upload an image to apply filters", type=["jpeg", "jpg", "png"])
    if input_image:
        user_image = load_image(input_image)
        st.image(user_image, caption=input_image.name, width=200)
        # saving image
        with open(os.path.join("pages/images", input_image.name), 'wb') as f:
            f.write(input_image.getbuffer())
        input_filter = "pages/images/"+input_image.name
        filter_input(input_filter)

############################ Default images ##################################
    st.write("### Image filtering with default images")
    with st.expander("View default input images"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_or[0], caption='Image view 1')
        with col2:
            st.image(img_or[1], caption="Image plan view")
        with col3:
            st.image(img_or[2], caption="Image vird view")
        col4, col5 = st.columns(2)

        with col4:
            st.image(img_or[3], caption="Image snake")
        with col5:
            st.image(img_or[4], caption='Image man')

    ################### Linear ###################
    #################### Mean ####################
    # st.write("### Image filtering results")

    with st.expander("Image filtering impementation results"):
        st.write("""
        
            The different filtering algorithms are categorized by columns: `Mean`, `Median` and `Sobel` filters. 
        """)

        image1 = cv2.imread(img_or[0])
        image2 = cv2.imread(img_or[1])
        image3 = cv2.imread(img_or[2])
        image4 = cv2.imread(img_or[3])
        image5 = cv2.imread(img_or[4])
        img_list = [image1, image2, image3, image4, image5]

        # mean filter function
        def meanFilter(images):
            filtered = [cv2.blur(i, (3,3)) for i in images]
            return filtered

        # median filter function
        def medianFilter(images):
            filtered = [cv2.medianBlur(i, 5) for i in images]
            return filtered
############################## SOBEL ##############################
        rgb_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]
        gray_imgs = [cv2.cvtColor(rgb_img , cv2.COLOR_BGR2GRAY) for rgb_img in rgb_imgs]
        gauss_blur = [cv2.GaussianBlur(gray_img, (5,5), 0) for gray_img in gray_imgs]

        # sobel function
        def sobel(gauss_blured):
            sobel_filtered = []
            for i in gauss_blured:
                x = cv2.Sobel(i, cv2.CV_16S, 1, 0)
                y = cv2.Sobel(i, cv2.CV_16S, 0, 1)
                absX = cv2.convertScaleAbs(x)
                absY = cv2.convertScaleAbs(y)

                sobel_filtered.append(cv2.addWeighted(absX, 0.5, absY, 0.5, 0))

            return sobel_filtered
##########################################################################################

        meanFiltered = meanFilter(img_list)
        medianFiltered = medianFilter(img_list)
        sobel_images = sobel(gauss_blur)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("__Original__")
            st.image(img_or[0], caption="Original image 1")
            st.image(img_or[1], caption="Original image 2")
            st.image(img_or[2], caption="Original image 3")
            st.image(img_or[3], caption="Original image 4")
            st.image(img_or[4], caption="Original image 5")

        with col2:
            st.write("__Mean Filter__")
            st.image(meanFiltered[0], caption="Mean filtered image 1")
            st.image(meanFiltered[1], caption="Mean filtered image 2")
            st.image(meanFiltered[2], caption="Mean filtered image 3")
            st.image(meanFiltered[3], caption="Mean filtered image 4")
            st.image(meanFiltered[4], caption="Mean filtered image 5")

        with col3:
            st.write("__Median Filter__")
            st.image(medianFiltered[0], caption="Median filtered image 1")
            st.image(medianFiltered[1], caption="Median filtered image 2")
            st.image(medianFiltered[2], caption="Median filtered image 3")
            st.image(medianFiltered[3], caption="Median filtered image 4")
            st.image(medianFiltered[4], caption="Median filtered image 5")

        with col4:
          st.write("__Sobel Filter__")
          st.image(sobel_images[0], caption="Sobel filtered image 1")
          st.image(sobel_images[1], caption="Sobel filtered image 2")
          st.image(sobel_images[2], caption="Sobel filtered image 3")
          st.image(sobel_images[3], caption="Sobel filtered image 4")
          st.image(sobel_images[4], caption="Sobel filtered image 5")

    