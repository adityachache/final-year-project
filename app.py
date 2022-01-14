from multiapp import Multiapp
from pages import home, filtering, compression, realtime
import streamlit as st
from streamlit.hashing import _CodeHasher

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx

    
app = Multiapp()

app.add_app("Home", home.app)
app.add_app("Object Detection", realtime.app)
app.add_app("Image Filtering", filtering.app)
app.add_app("Image Compression", compression.app)

app.run()