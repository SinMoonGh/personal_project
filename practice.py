import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
     
scaled_data = pd.read_csv('scaled_data.csv', encoding='cp949')

# 전류 흐름도
electric= []
for weld_t in scaled_data['전류']:
    electric.append(weld_t)

# 데이터 생성
# y가 전류가 흐르는 모양으로 가면 될 듯
data = pd.DataFrame({'x': range(10000), 'y': electric[:10000]})

# streamlit 앱 설정
st.title('동적 그래프 예제')
slider = st.slider('데이터 범위 선택', 0, len(data) - 100, 0, 100)
plt.plot(data['x'][slider:slider+100], data['y'][slider:slider+100])
plt.ylim(0, 0.001)
st.pyplot(plt)

