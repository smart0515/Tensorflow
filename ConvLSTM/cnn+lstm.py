from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense

# 층을 구성하는 모델을 생성합니다.
model = Sequential()

# 1D Convolutional Layer를 추가합니다.
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))

# Max Pooling Layer를 추가합니다.
model.add(MaxPooling1D(pool_size=2))

# Flatten Layer를 추가합니다.
model.add(Flatten())

# LSTM Layer를 추가합니다.
model.add(LSTM(100, activation='relu'))

# Dense Layer를 추가합니다.
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 훈련합니다.
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
