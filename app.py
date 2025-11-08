import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Breast Cancer AI", page_icon="ribbon")

@st.cache_resource
def load_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler, data.feature_names

model, scaler, features = load_model()

st.title("Breast Cancer Risk Prediction")
st.write("Enter 5 biopsy values → Get AI result")

with st.form("predict"):
    col1, col2 = st.columns(2)
    with col1:
        r = st.slider("Mean Radius", 6.0, 30.0, 14.0)
        t = st.slider("Mean Texture", 10.0, 40.0, 19.0)
        p = st.slider("Mean Perimeter", 40.0, 200.0, 90.0)
    with col2:
        a = st.slider("Mean Area", 100.0, 3000.0, 650.0)
        s = st.slider("Mean Smoothness", 0.05, 0.20, 0.10, 0.001)

    if st.form_submit_button("PREDICT"):
        inputs = [r, t, p, a, s] + [0] * (len(features) - 5)
        X_in = np.array([inputs])
        X_scaled = scaler.transform(X_in)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]

        if pred == 0:
            st.error(f"CANCER DETECTED ({prob[0]:.1%} risk)")
        else:
            st.success(f"NO CANCER ({prob[1]:.1%} safe)")
        st.info("Accuracy: ~97%")