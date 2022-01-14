import streamlit as st
import numpy as np 
import pandas as pd 
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt

def app():
    # load images function
    def load_image(img):
        out_img = Image.open(img)
        return out_img

    st.write("# Image Compression")
    st.write("""
    Image compression is an application of data compression that encode the original image with few bits. 
    It is basically an approach to represent and store information about the images in a minimum number of bits without losing the character of the image.

    In this section we want to compress images using the `JPEG` standard.

    
    """)
    
    st.write("### Image compression with user input")
    toCompress = st.file_uploader("Upload the image to compress", type=["jpeg", "jpg", "png"])
    if toCompress:
        img_to_compress = load_image(toCompress)
        
        # save image
        with open(os.path.join("pages/images", toCompress.name), "wb") as f:
            f.write(toCompress.getbuffer())
            st.caption("Image saved")

        loc_img = "pages/images/"+toCompress.name
        orig_img_size = (os.stat(loc_img).st_size)*0.001
        cap_original = "Original image size: "+str(orig_img_size)+" KB"
        # st.image(img_to_compress, caption=cap_original, width=200)

        st.write("""
            After uploading the image, you will have to specify three different image qualities 
            to compress the original image `(0 <= Quality <= 100)`.
            """)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            imgQuality4 = st.number_input("Original Image", min_value=100, max_value=100, key=1)
            
            st.image(img_to_compress, caption=cap_original)

        with col2:
            imgQuality1 = st.number_input("Compression quality 1", min_value=0, max_value=100, key=2)
            img1 = "pages/images/Compressed1_"+toCompress.name
            img_to_compress.save(img1, optimize=True, quality=imgQuality1)
            sizekb1 = (os.stat(img1).st_size)*0.001
            cap1 = "Compressed image 1 size: "+str(sizekb1)+" KB"
            st.image(img1, caption=cap1)

        with col3:
            imgQuality2 = st.number_input("Compression quality 2", min_value=0, max_value=100, key=3)
            img2 = "pages/images/Compressed2_"+toCompress.name
            img_to_compress.save(img2, optimize=True, quality=imgQuality2)
            sizekb2 = (os.stat(img2).st_size)*0.001
            cap2 = "Compressed image 2 size: "+str(sizekb2)+" KB"
            st.image(img2, caption=cap2)

        with col4:
            imgQuality3 = st.number_input("Compression quality 3", min_value=0, max_value=100, key=4)
            img3 = "pages/images/Compressed3_"+toCompress.name
            img_to_compress.save(img3, optimize=True, quality=imgQuality3)
            sizekb3 = (os.stat(img3).st_size)*0.001
            cap3 = "Compressed image 3 size: "+str(sizekb3)+" KB"
            st.image(img3, caption=cap3)
        
        
            
        st.write("### JPEG Standard")
        st.write("""
    

    """)
    st.image('images/compression.PNG', caption="The flow of image compression coding")

    st.write("""
        To compress the input image, we have to follow these steps:

        1. `Y, Cb, Cr conversion` 
        2. `Divide the image to 8x8 blocks`
        3. `Apply DCT to each block`
        4. `Quantization` 
        5. `Inverse DCT`
    """)

    # # Take input image
    # image_jpeg = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    # if image_jpeg:
    #     st.image(image_jpeg, caption="")

    


################ Quantization Arrays ################

    def selectQMatrix(qName):
        Q10 = np.array([[80,60,50,80,120,200,255,255],
                    [55,60,70,95,130,255,255,255],
                    [70,65,80,120,200,255,255,255],
                    [70,85,110,145,255,255,255,255],
                    [90,110,185,255,255,255,255,255],
                    [120,175,255,255,255,255,255,255],
                    [245,255,255,255,255,255,255,255],
                    [255,255,255,255,255,255,255,255]])

        Q50 = np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,130,99]])

        Q90 = np.array([[3,2,2,3,5,8,10,12],
                        [2,2,3,4,5,12,12,11],
                        [3,3,3,5,8,11,14,11],
                        [3,3,4,6,10,17,16,12],
                        [4,4,7,11,14,22,21,15],
                        [5,7,11,13,16,12,23,18],
                        [10,13,16,17,21,24,24,21],
                        [14,18,19,20,22,20,20,20]])
        if qName == "Q10":
            return Q10
        elif qName == "Q50":
            return Q50
        elif qName == "Q90":
            return Q90
        else:
            return np.ones((8,8)) 

    # display y,cb,cr
    def showImage(img):
        plt.figure(figsize=(8,8))
        plt.imshow(img,cmap=None)
        plt.xticks([]),plt.yticks([])
        plt.show()
    
################################################################
    with st.expander("JPEG work flow with default image"):
        default_image = 'images/189080.jpg'
        img_ycbcr = cv2.imread(default_image, 0)
        col1, col2 = st.columns(2)
        with col1:
            st.write("__Default original Image__")
            st.image(default_image, caption="Original Image")
        with col2:
            st.write("__Y Component__")
            fig = plt.figure()
            plt.imshow(img_ycbcr, cmap=None)
            st.pyplot(fig)
            # st.image(img_ycbcr, caption="Y component")

        st.write("#### 8x8 block Splitting")
        height  = len(img_ycbcr) #one column of image
        width = len(img_ycbcr[0]) # one row of image
        sliced = [] # new list for 8x8 sliced image 
        block = 8
            
        ## divide to 8x8 blocks
        #dividing 8x8 parts
        currY = 0 #current Y index
        for i in range(block,height+1,block):
            currX = 0 #current X index
            for j in range(block,width+1,block):
                sliced.append(img_ycbcr[currY:i,currX:j]-np.ones((8,8))*128) #Extracting 128 from all pixels
                currX = j
            currY = i
            
        
        output_size = "The image height is " +str(height)+", and image width is "+str(width)+" pixels"
        st.write(output_size)
        st.write('We want to divide our image to 8x8 blocks, on which we will be applying the `Discrete Cosine Transform`')
        sliced_size = "Size of the sliced image: "+str(len(sliced))
        sliced_elem_size = "Each elemend of sliced list contains a "+ str(sliced[0].shape)+ " element."
        st.write(sliced_size)
        st.write(sliced_elem_size)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.image(sliced[0].astype(int), caption="1st 8x8 block")
        with col2:
            st.image(sliced[1].astype(int), caption="2nd 8x8 block")
        with col3:
            st.image(sliced[2].astype(int), caption="3rd 8x8 block")
        with col4:
            st.image(sliced[3].astype(int), caption="4th 8x8 block")
        with col5:
            st.image(sliced[4].astype(int), caption="5th 8x8 block")
        with col6:
            st.image(sliced[5].astype(int), caption="6th 8x8 block")

        imf = [np.float32(img) for img in sliced]

        #################### DCT ####################

        DCToutput = []
        for part in imf:
            currDCT = cv2.dct(part)
            DCToutput.append(currDCT)
        
        #### Quantization ####
        #la plupart des éléments de fréquence supérieure du sous-bloc  sont compressés en valeurs nulles.
        #As we can see quantization is completed. The size has been recuded
        selectedQMatrix = selectQMatrix("Q90")
        for ndct in DCToutput:
            for i in range(block):
                for j in range(block):
                    #
                    ndct[i,j] = np.around(ndct[i,j]/selectedQMatrix[i,j])
        
        # inverse DCT
        invList = []
        for ipart in DCToutput:
            ipart
            curriDCT = cv2.idct(ipart)
            invList.append(curriDCT)

        row = 0
        rowNcol = []
        for j in range(int(width/block),len(invList)+1,int(width/block)):
            rowNcol.append(np.hstack((invList[row:j])))
            row = j
        res = np.vstack((rowNcol))

        st.write('__Compressed Image__')
        st.write("""
            After Applying Discrete Cosine Transform, Quantization and Inverse DCT, we can see the compressed image shown below.
        """)
        fig = plt.figure()
        plt.imshow(res, cmap=None)
        st.pyplot(fig)

