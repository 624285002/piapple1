import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle

st.image('./pic/welcome.jpg')

html_8="""
<div style="background-color:#228B22;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายโรคสับปะรด</h5></center>
</div>
"""

st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv("./data/piapple.csv")
st.write(dt.head(10))
dt1 = dt['tip.soil'].sum()
dt2 = dt['leaf.width'].sum()
dx=[dt1,dt2]
dx2=pd.DataFrame(dx,index=["d1","d2"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.area_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")
st.sidebar.markdown("# วิเคราห์รายบุคคล")

html_8="""
<div style="background-color:#228B22;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""

st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

pi_soil=st.number_input("กรุณาเลือกข้อมูล ความสูงจากยอดใบล่างถึงพื้น")
pi_wid=st.number_input("กรุณาเลือกข้อมูล ความกว้างของใบ")

if st.button("ทำนายผล"):
    loaded_model = pickle.load(open('./data/piapple_model.sav', 'rb'))
    input_data =  (pi_soil,pi_wid)
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

    st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")


