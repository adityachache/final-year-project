import streamlit as st
import PIL.Image as Image

    
def app():
    st.title("Welcome To Our App")

    st.subheader("This app aims to provide a one stop solution for all media processing")

    image11 = Image.open("pages/pic11.jpg")

    st.image(image11,use_column_width=True)

    st.subheader("""The App Consists of Two Modules:""")

    st.title("Module 1:Image Processing")

    st.write("""This consists of image filtering and image compression, this aims to provide the platform and functionality to process the media before using it for detection purpose""")

    image12 = Image.open("pages/pic12.jpg")

    st.image(image12,use_column_width=True)


    st.title("Module 2:Object Detection")

    st.write("""This provides the user to choose the type of media he/she wants to work with it can be
    1. Images
    2. Video Streaming
    """)

    image13 = Image.open("pages/pic13.jpg")

    st.image(image13,use_column_width=True)


    st.write ("It also includes a work flow section which explains the main algorithm behind the Object Detection i.e. YOLO which will be helpful for 1st time learners")


app()
