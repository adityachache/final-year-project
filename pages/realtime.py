
#importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import urllib

import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

    
import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
from aiortc.contrib.media import MediaPlayer
import streamlit_webrtc
from streamlit_webrtc import AudioProcessorBase, RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

    
HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

def app():
    # This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
    def download_file(url, download_to: Path, expected_size=None):
        # Don't download the file twice.
        # (If possible, verify the download using the file length.)
        if download_to.exists():
            if expected_size:
                if download_to.stat().st_size == expected_size:
                    return
            else:
                st.info(f"{url} is already downloaded.")
                if not st.button("Download again?"):
                    return

        download_to.parent.mkdir(parents=True, exist_ok=True)

        # These are handles to two visual elements to animate.
        weights_warning, progress_bar = None, None
        try:
            weights_warning = st.warning("Downloading %s..." % url)
            progress_bar = st.progress(0)
            with open(download_to, "wb") as output_file:
                with urllib.request.urlopen(url) as response:
                    length = int(response.info()["Content-Length"])
                    counter = 0.0
                    MEGABYTES = 2.0 ** 20.0
                    while True:
                        data = response.read(8192)
                        if not data:
                            break
                        counter += len(data)
                        output_file.write(data)

                        # We perform animation by overwriting the elements.
                        weights_warning.warning(
                            "Downloading %s... (%6.2f/%6.2f MB)"
                            % (url, counter / MEGABYTES, length / MEGABYTES)
                        )
                        progress_bar.progress(min(counter / length, 1.0))
        # Finally, we remove these visual elements by calling .empty().
        finally:
            if weights_warning is not None:
                weights_warning.empty()
            if progress_bar is not None:
                progress_bar.empty()


    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )




    #function for object detection
    def obj_detection():
        
        st.title('Object Detection-Images')
        st.subheader("This app takes the input as image and outputs the image with detected objects with their confidence score.")

        uploaded_file = st.file_uploader("Upload a image",type='jpg')
        if uploaded_file != None:
            image1 = Image.open(uploaded_file)
            image2 =np.array(image1)
            
            st.image(image1, caption='Uploaded Image.')
            
            my_bar = st.progress(0)
            
            confThreshold =st.slider('Confidence', 0, 100, 50)
            nmsThreshold= st.slider('Threshold', 0, 100, 20)
            whT = 320
            #### LOAD MODEL
            ## Coco Names
            classesFile = "pages/coco.names"
            classNames = []
            with open(classesFile, 'rt') as f:
                classNames = f.read().split('\n')
                
            
            ## Model Files        
            modelConfiguration = "pages/yolov3-tiny.cfg"
            modelWeights = "pages/yolov3-tiny.weights"
            net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            #finding the objects
            def findObjects(outputs,img):
                hT, wT, cT = image2.shape
                bbox = []
                classIds = []
                confs = []
                for output in outputs:
                    for det in output:
                        scores = det[5:]
                        classId = np.argmax(scores)
                        confidence = scores[classId]
                        if confidence > (confThreshold/100):
                            w,h = int(det[2]*wT) , int(det[3]*hT)
                            x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                            bbox.append([x,y,w,h])
                            classIds.append(classId)
                            confs.append(float(confidence))
            
                indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold/100, nmsThreshold/100)
                obj_list=[]
                confi_list =[]
                #drawing rectangle around object
                for i in indices:
                    # i = i[0]
                    box = bbox[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    # print(x,y,w,h)
                    cv2.rectangle(image2, (x, y), (x+w,y+h), (255, 0 , 255), 2)
                    #print(i,confs[i],classIds[i])
                    obj_list.append(classNames[classIds[i]].upper())
                    
                    confi_list.append(int(confs[i]*100))
                    cv2.putText(image2,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
                if st.checkbox("Show Object's list" ):
                    
                    st.write(df)
                if st.checkbox("Show Confidence bar chart" ):
                    st.subheader('Bar chart for confidence levels')
                    
                    st.bar_chart(df["Confidence"])
               
            blob = cv2.dnn.blobFromImage(image2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(outputNames)
            findObjects(outputs,image2)
        
            st.image(image2, caption='Proccesed Image.')
            
            # cv2.waitKey(0)
            
            # cv2.destroyAllWindows()
            my_bar.progress(100)
            
            

    def app_object_detection():
        
        
        st.title('Object Detection-Images')
        st.subheader("This app takes the input from your webcam or video streaming device and detects objects in real time.")
        
        st.write("Select your Device for Video Streaming and Adjust Threshold Value as per need.")
        
        """Object detection demo with MobileNet SSD.
        """
        MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
        MODEL_LOCAL_PATH = HERE / "pages/models/MobileNetSSD_deploy.caffemodel"
        PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
        PROTOTXT_LOCAL_PATH = HERE / "pages/models/MobileNetSSD_deploy.prototxt.txt"

        CLASSES = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
        download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

        DEFAULT_CONFIDENCE_THRESHOLD = 0.3

        class Detection(NamedTuple):
            name: str
            prob: float

        class MobileNetSSDVideoProcessor(VideoProcessorBase):
            confidence_threshold: float
            result_queue: "queue.Queue[List[Detection]]"

            def __init__(self) -> None:
                self._net = cv2.dnn.readNetFromCaffe(
                    str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
                )
                self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
                self.result_queue = queue.Queue()

            def _annotate_image(self, image, detections):
                # loop over the detections
                (h, w) = image.shape[:2]
                result: List[Detection] = []
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > self.confidence_threshold:
                        # extract the index of the class label from the `detections`,
                        # then compute the (x, y)-coordinates of the bounding box for
                        # the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        name = CLASSES[idx]
                        result.append(Detection(name=name, prob=float(confidence)))

                        # display the prediction
                        label = f"{name}: {round(confidence * 100, 2)}%"
                        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(
                            image,
                            label,
                            (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            COLORS[idx],
                            2,
                        )
                return image, result

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                image = frame.to_ndarray(format="bgr24")
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
                )
                self._net.setInput(blob)
                detections = self._net.forward()
                annotated_image, result = self._annotate_image(image, detections)

                # NOTE: This `recv` method is called in another thread,
                # so it must be thread-safe.
                self.result_queue.put(result)

                return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=MobileNetSSDVideoProcessor,
            async_processing=True,
        )

        confidence_threshold = st.slider(
            "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
        )
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

        if st.checkbox("Show the detected labels", value=True):
            if webrtc_ctx.state.playing:
                labels_placeholder = st.empty()
                # NOTE: The video transformation with object detection and
                # this loop displaying the result labels are running
                # in different threads asynchronously.
                # Then the rendered video frames and the labels displayed here
                # are not strictly synchronized.
                while True:
                    if webrtc_ctx.video_processor:
                        try:
                            result = webrtc_ctx.video_processor.result_queue.get(
                                timeout=1.0
                            )
                        except queue.Empty:
                            result = None
                        labels_placeholder.table(result)
                    else:
                        break

        st.markdown("!!!"
        )
        
    def work_flow():
        st.title("Now How does YOLO Work?")


    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
            ["Object detection-Images","Real Time Detection","Work Flow"])
     
    if app_mode == "Object detection-Images":
        obj_detection()
         
    if app_mode == "Real Time Detection":
        app_object_detection()
        
    if app_mode == "Work Flow":
        work_flow()
        
        logger.debug("=== Alive threads ===")
        for thread in threading.enumerate():
            if thread.is_alive():
                logger.debug(f"  {thread.name} ({thread.ident})")

