# 필요한 라이브러리들을 임포트
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


#모델 구성, 훈련, 컴파일, 시각화가 이루어지는 파일


# 이미지 전처리를 위한 ImageDataGenerator 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# train 데이터셋과 validation 데이터셋을 로드
train_generator = train_datagen.flow_from_directory(
    'data/train',  # 이 경로는 train 이미지가 있는 폴더를 가리킴
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation', # 이 경로는 validation 이미지가 있는 폴더를 가리킴
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# CNN 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(
    train_generator,
    steps_per_epoch=100, # 총 훈련 샘플 수 / 배치 크기
    epochs=15,
    validation_data=validation_generator, # 검증 데이터 제너레이터
    validation_steps=50, # 총 검증 샘플 수 / 배치 크기
    verbose=2
)

# 훈련 과정 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# 모델 저장
model.save('models/cat_dog_classifier.h5')
