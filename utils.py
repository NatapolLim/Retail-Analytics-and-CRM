import pandas as pd
import streamlit as st

def load_css() -> None:
    '''open css file and activate its'''
    with open("style.css", 'r') as file:
        st.markdown("<style>{}</style>".format(file.read()), unsafe_allow_html=True)