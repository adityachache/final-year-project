import streamlit as st
from multiapp import Multiapp
from pages import home, filtering, compression, realtime
# try:
#     import streamlit.ReportThread as ReportThread
#     from streamlit.server.Server import Server
# except Exception:
#     # Streamlit >= 0.65.0
#     import streamlit.report_thread as ReportThread
#     from streamlit.server.server import Server

app = Multiapp()

app.add_app("Home", home.app)
app.add_app("Object Detection", realtime.app)
app.add_app("Image Filtering", filtering.app)
app.add_app("Image Compression", compression.app)

app.run()