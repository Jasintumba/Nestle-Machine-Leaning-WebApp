import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



st.set_page_config(
page_title="Multipage App",
page_icon="👋",
)

st.title("Nestle Online Processing Dashboard")
st.sidebar.success("Select Specific Processing Plant above.")

st.write("Note: This application is for authorized personnel only.")

st.subheader(f'CURRENT PROCESSING PARAMETERS')
st.write('Mono Pump Speed:', 35)

st.write('Mono Pump Flowrate:', 2360)

st.write('DSI Valve Opening:', 45)

st.write('DSI 1 Temperature:', 75)

st.write('Hydrolisis Waukesha Pump speed:', 85)

st.write('DSI 2 Valve opening:', 61)

st.write('DSI  2 Temperature:', 134)

st.write('CCP 1 Temperature:', 134)

st.write('CCP 2 Temperature:', 131)

st.write('Flash Tank Waukesha Speed:', 100)