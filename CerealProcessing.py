import streamlit as st 
import numpy as np 
import time
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random as rnd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title('Nestle Online Processing CockPit')

st.write("""
# Optimize Process control input parameters
Compare process value and optimized value?
""")

dataset_name = st.sidebar.selectbox(
    'Select Processing Area',
    ('Cereals Processing', 'Milks processing', 'Culinary Processing')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select Optimization Algorithm',
    ('Cereals Optimizer', 'Milks Optimizer', 'Random Forest')
)

def get_dataset(name):
    data = None
    if name == 'Cereals Processing':
        data = datasets.load_iris()
    elif name == 'Milks processing':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Mono Pump Speed:', 35, 'Optimized Speed:', rnd.randint(30,40))
st.write('Mono Pump Flowrate:', 2360, 'Optimized Pump flowrate:', rnd.randint(2000,2800 ))
st.write('DSI Valve Opening:', 45, 'Optimized DSI1 Opening:', rnd.randint(40,50 ))
st.write('DSI 1 Temperature:', 75, 'Optimized DSI 1 temperature:', rnd.randint(70,80 ))
st.write('Hydrolisis Waukesha Pump speed:', 85, 'Optimized Hydrolysis Wuukesha Speed:', rnd.randint(80,100 ))
st.write('DSI 2 Valve opening:', 61, 'Optimized DSI 2 Opening:', rnd.randint(60,80 ))
st.write('DSI  2 Temperature:', 134, 'Optimized DSI 2 Tempereture:', rnd.randint(130,135 ))
st.write('CCP 1 Temperature:', 134, 'Optimized CCP 1 Tempereture:', rnd.randint(130,135 ))
st.write('CCP 2 Temperature:', 131, 'Optimized CCP 2 Tempereture:', rnd.randint(130,135 ))
st.write('Flash Tank Waukesha Speed:', 100, 'SOUP SUPPLY PUMP SPEED:', rnd.randint(90,100 ))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Milks Optimizer':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'Cereals Optimizer':
        K = st.sidebar.slider('Number of processs input parameters', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'Milks Optimizer':
        clf = SVC(C=params['C'])
    elif clf_name == 'Cereals Optimizer':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc )
st.write(f'OPTIMIZED PROCESS PARAMETERS')

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
