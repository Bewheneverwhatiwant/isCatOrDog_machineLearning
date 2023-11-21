from keras.layers import Flatten, Dense

#이미지 분류 모델

# 특징 추출 부분 뒤에 완전 연결층 추가
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 활성화 함수
