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

dt=pd.read_csv("./data/Smote.csv")
st.write(dt.head(9))
dt1 = dt['age'].sum()
dt2 = dt['objective'].sum()
dt3 = dt['price'].sum()
dt4 = dt['store'].sum()
dt5 = dt['motivation'].sum()
dt6 = dt['parent_income'].sum()
dx=[dt1,dt2,dt3,dt4,dt5,dt6]
dx2=pd.DataFrame(dx,index=["d1","d2","d3","d4","d5","d6"])


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

age=st.number_input("กรุณากรอกข้อมูล ความสูงจากยอดใบล่างถึงพื้น")
objective=st.number_input("กรุณากรอกข้อมูล ความกว้างของใบ")
price=st.number_input("กรุณากรอกข้อมูล ความกว้างของใบ")
store=st.number_input("กรุณากรอกข้อมูล ความกว้างของใบ")
motivation=st.number_input("กรุณากรอกข้อมูล ความกว้างของใบ")
parent_income=st.number_input("กรุณากรอกข้อมูล ความกว้างของใบ")

if st.button("ทำนายผล"):
    loaded_model = pickle.load(open('./data/smote_model.sav', 'rb'))
    input_data =  (age,objective,price,store,motivation,parent_income)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    st.write(prediction)
    if prediction == 'top rot':
        st.image('./pic/top rot.jpg')
    elif prediction == 'withered':
        st.image('./pic/withered.jpg')
    elif prediction == 'withered':
        st.image('./pic/withered.jpg')
    elif prediction == 'withered':
        st.image('./pic/withered.jpg')
    elif prediction == 'withered':
        st.image('./pic/withered.jpg')
    else:
        st.image('./pic/normal.jpg')
else:
    st.write("")


