import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

# b = [15, 15, 25, 13, 17, 15, 35]
# for index, values in enumerate(welding_data.columns[3:]):
#     print(index, values)


# welding Data 이용
welding_data = pd.read_excel('welding_data.xlsx', index_col = 'idx')


new_welding_data = welding_data.iloc[:, 5:]

scaler = preprocessing.MinMaxScaler()
scaler_fit = scaler.fit(new_welding_data)
print(scaler_fit)
