from keras.preprocessing.image import ImageDataGenerator

#이미지 전처리 모델

train_datagen = ImageDataGenerator(
    rescale=1./255,    # 정규화
    rotation_range=40, # 이미지 회전
    width_shift_range=0.2, # 가로로 이동
    height_shift_range=0.2, # 세로로 이동
    shear_range=0.2,   # 전단 변환
    zoom_range=0.2,    # 확대
    horizontal_flip=True, # 수평 뒤집기
    fill_mode='nearest' # 회전 또는 이동 후 빈 공간 채우기
)

# train 데이터셋을 'data/train' 경로에서 로드
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150), # 모든 이미지를 150x150 크기로 조정
    batch_size=32,
    class_mode='binary' # binary_crossentropy 손실을 사용하기 때문에 이진 레이블 필요
)
