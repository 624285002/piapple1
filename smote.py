import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle


st.image('./pic/piapple.jpg')

html_8="""
<div 
            style="background-color:orange;
            padding:5px;
            border-radius:0px 0px 0px 0px;
            border-style:'solid';
            border-color:white">
<center><h3>การทำนายโรคสับปะรด ด้วยเทนนิค KNN</h3></center>
</div>
"""

st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv("./data/piapple.csv")
st.write(dt.head(9))
dt1 = dt['tip.soil'].sum()
dt2 = dt['leaf.width'].sum()
dx=[dt1,dt2]
dx2=pd.DataFrame(dx,index=["d1","d2"])


html_8="""
<div style="background-color:orange;
            padding:5px;
            border-radius:0px 0px 0px 0px;
            border-style:'solid';
            border-color:white">
<center><h3>กรอกข้อมูลเพื่อทำนายโรค</h3></center>
</div>
"""

st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

tip_soil=st.number_input("กรุณากรอกข้อมูล ความสูงจากยอดใบล่างถึงพื้น")
leaf_width=st.number_input("กรุณากรอกข้อมูล ความกว้างของใบ")

if st.button("ทำนายผล"):
    loaded_model = pickle.load(open('./data/piapple_model.sav', 'rb'))
    input_data =  (tip_soil,leaf_width)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    st.write(prediction)
    if prediction == 'top rot':
        st.image('./pic/top rot.jpg')
    elif prediction == 'withered':
        st.image('./pic/withered.jpg')
    else:
        st.image('./pic/normal.jpg')
else:
    st.write("")


