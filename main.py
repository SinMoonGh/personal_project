import streamlit as st
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import numpy as np
import pandas as pd
import sklearn.metrics as metric
from sklearn import preprocessing
import torch.nn as nn
import matplotlib.pyplot as plt


"""데이터 불러오기 -> 
데이터 종류 및 개수 확인 -> 
데이터 특성 파악(요약 통계량, 상관관계, histogram) ->
데이터 정제(전처리) (변수제거, 정규화) ->
오토인코더 모델 구축(잡음제거) ->
훈련 데이터/ 테스트 데이터 분할 ->
초매개변수, 손실 함수 및 옵티마이저 정의 ->
오토인코더 학습 함수 정의 및 학습 ->
임계값 정의 후 결과 분석 및 해석
"""

# TODO 1: 라이브러리/ 데이터 불러오기

# welding_data.xlsx 파일을 불러옵니다.
welding_data = pd.read_excel('welding_data.xlsx', index_col = 'idx')
welding_data.head()

# TODO 2: 데이터 종류 및 개수 확인

# 데이터의 특성들이 어떤 값들을 가지고 있으며 몇 개씩 가지고 있는 지 확인
for feature in welding_data:
    print(feature, welding_data[feature].value_counts())

# 데이터 특성 파악
welding_data.describe()

# 용접 제품/corr 함수를 통한 변수 간 상관관계 파악 가이드
welding_data.corr(numeric_only=True)

# 용접 제품/ histogram을 통한 변수별 데이터 파악 가이드
b = [15, 15, 25, 13, 17, 15, 35]

thickness_1 = plt.hist(welding_data['Thickness 1(mm)'], bins = b[0], facecolor = (144 /255,171 /255,221 /255), range=(0, 1.2))
st.write()

thickness_2 = plt.hist(welding_data['Thickness 2(mm)'], bins = b[1], facecolor = (144 /255,171 /255,221 /255), range=(0, 1.2))
st.write()

weld_force = plt.hist(welding_data['weld force(bar)'], bins = b[2], facecolor = (144 /255,171 /255,221 /255), range=(0, 120))
st.write()

weld_current = plt.hist(welding_data['weld current(kA)'], bins = b[3], facecolor = (144 /255,171 /255,221 /255), range=(0, 160))
st.write()

weld_Voltage = plt.hist(welding_data['weld Voltage(v)'], bins = b[4], facecolor = (144 /255,171 /255,221 /255), range=(0, 200))
st.write()

weld_time = plt.hist(welding_data['weld time(ms)'], bins = b[5], facecolor = (144 /255,171 /255,221 /255), range=(0, 12000))
st.write()

# TODO 3: 데이터 정제(전처리)
# 데이터 품질 검사, 데이터 정제(전처리) & 가공 실습 내용

# 용접기 데이터에서 필요없는 부분(생산 품목, 작업 시간, 소재두께)들을 제외
new_welding_data = welding_data.iloc[:, 5:]
new_welding_data.head()

# MinMaxScaler를 통한 데이터 정규화 가이드

# sklearn의 preprocessing 모듈에 들어있는 MinMaxScaler함수를 이용해 정규화 적용
scaler = preprocessing.MinMaxScaler()
scaler.fit(new_welding_data)
scaled_data = scaler.transform(new_welding_data)
scaled_data

# AutoEncoder 클래스 구현
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        # initialize
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 오토인코더 구현
        self.AutoEncoder = nn.Sequential(
            #인코더 
            nn.Linear(input_size, hidden_size[0]),
            nn.RReLU(),
            nn.Linear(hidden_size[0], output_size),
            nn.RReLU(),
            # 디코더
            nn.Linear(output_size, hidden_size[0]),
            nn.RReLU(),
            nn.Linear(hidden_size[0], input_size)
        )

    def forward(self, inputs):
        output = self.AutoEncoder(inputs)

        return output
    
# 훈련 데이터/ 테스트 데이터 분할

# 기존이 데이터를 텐서 형태로 변환, 그리고 훈련세트와 테스트세트로 나눔
train_data = torch.Tensor(scaled_data[:8470]) # 처음부터 8469번까지 데이터를 훈련세트로 지정
print(len(train_data))
test_data = torch.Tensor(scaled_data[8479:]) # 8470번째 데이터부터 끝까지를 테스트세트로 지정
print(len(test_data))

# 하이퍼파라미터, 손실 함수 및 옵티마이저 정의

# 훈련 하이퍼 파라미터
epoch = 50
batch_size = 64
lr = 0.01
# 모델 하이퍼 파라미터
input_size = len(train_data[0])
hidden_size = [3]
output_size = 2
# 손실 함수로 제곱근 오차 사용
criterion = nn.MSELoss()
# 매개 변수 조정 방식으로 Adam 사용
optimizer = torch.optim.Adam
#오토인코더 정의
AutoEncoder = AutoEncoder(input_size, hidden_size, output_size)

# 오토인코더 학습 함수 정의 및 학습

# 학습 함수에 대한 정의
def train_net(AutoEncoder, data, criterion, epochs, lr_rate = 0.01):
    # Optimizer에 대한 정의
    optim = optimizer(AutoEncoder.parameters(), lr = lr_rate)
    #배치 학습을 시키기 위한 데이터 변환
    data_iter = DataLoader(data, batch_size = batch_size, shuffle=True)
    #에포크 학습
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for x in data_iter:
            # 매개변수 0으로 초기화
            optim.zero_grad()
            output = AutoEncoder(x)
            # 입력값과 출력값 간의 차이인 손실값
            loss = criterion(x, output)
            # 손실 값을 기준으로 매개변수 조정
            loss.backward()
            optim.step()
            running_loss += loss.item()

        # 각 에포크마다 손실 값 표기
        print("epoch: {}, loss: {:.2f}".format(epoch, running_loss))
    return AutoEncoder


# 오토인코더 학습 과정 가이드
# AI 분석 모델 학습 내용(학습 -> 검증 -> 평가 비유르 학습 조건, 모델 튜닝)

# 학습 함수를 이용한 오토인코더 학습
AutoEncoder = train_net(AutoEncoder, train_data, criterion, epoch, lr)

# 임계값 정의 후 결과 분석 및 해석

# 훈련세트의 손실 값 이용한 임계값 정의
train_loss_chart = []
for data in train_data:
    output = AutoEncoder(data)
    loss = criterion(output, data)
    train_loss_chart.append(loss.item())

threshold = np.mean(train_loss_chart) + np.std(train_loss_chart)*8
print("Threshold : ", threshold)

# 분석 결과값 도출

# 훈련 세트의 손실값 이용한 임계값 정의
test_loss_chart = []
for data in train_data:
    output = AutoEncoder(data)
    loss = criterion(output, data)
    test_loss_chart.append(loss.item())

outlier = list(test_loss_chart >= threshold)
outlier.count(True) # 이번 테스트에서는 20개의 불량품이 나온 것을 알 수 있다.



# 전류 그래프

# 그래프를 그릴 때 100개 씩만 해서 시간에 따라 변하는 그래프 만들기

# # 용접 가압력, 전류 산점도
# chat_data_1 = scaled_data[['용접 가압력', '전류']]
# st.scatter_chart(chat_data_1)


# 슬라이드 동적으로

# 0.1 초마다 그래프 바뀌게 for문으로