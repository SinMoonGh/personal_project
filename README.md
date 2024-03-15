## MinMaxScaler를 통한 데이터 정규화 가이드

- scaler = preprocessing.MinMaxScaler() : 각 특성을 지정된 범위로 확장하여 특성을 변환합니다.
- fit() : 스케일일에 사용할 최소값과 최대값을 계산합니다.
- transform() : 입력된 값을 변환합니다.
- 각 특성을 MinMaxScaler 함수를 사용하여 최소-최대 정규화를 적용했다.

## 오토인코더 모델 구축(잡음제거)

- ai 분석 모델 구축을 위한 방법론(알고리즘) 적용 및 학습 네트워크 구축 실습

| input_size | 입력의 크기, 숫자로 입력을 받음 |
| hidden_size | 신경망층의 크기, 리스트 형태로 입력을 받음 |
| output_size | 출력의 크기, 숫자로 입력을 받음 |
| nn.linear() | 완전연결망 |
| nn.RReLU() | Leaky Rectified Linear Unit 활성 함수 |
| nn.RReLU() | 인공신경망을 담는 컨테이너, 여기에 각 신경망 층들을 더해주어 신경망을 만듦 |

### 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 용접기 AI 데이터셋, KAIST(울산과학기술원, ㈜이피엠솔루션즈), 2020.12.14., www.kamp-ai.kr
