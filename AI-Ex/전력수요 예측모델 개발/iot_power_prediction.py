
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Flatten
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Add

# 데이터 로드 및 전처리
file_path = 'household_power_consumption.txt'
df = pd.read_csv(
    file_path,
    sep=';',
    parse_dates={'Datetime': ['Date', 'Time']},
    infer_datetime_format=True,
    na_values='?',
    low_memory=False
)
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df.dropna(inplace=True)
df.set_index('Datetime', inplace=True)

# 일 단위 리샘플링
daily_power = df['Global_active_power'].resample('D').mean()
daily_power.dropna(inplace=True)

# Train/Test 분할
total_days = len(daily_power)
train_size = int(total_days * 0.8)
train = daily_power.iloc[:train_size]
test = daily_power.iloc[train_size:]

# 시퀀스 생성 함수
def create_sequences(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        seq_x = series[i:i + window_size]
        seq_y = series[i + window_size]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# 시퀀스 생성 및 정규화
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))
window_size = 7
X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)
X_train = X_train.reshape((X_train.shape[0], window_size, 1))
X_test = X_test.reshape((X_test.shape[0], window_size, 1))

# 공통 평가 함수
def evaluate_model(y_test, y_pred, scaler, name="Model"):
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    mask = ~np.isnan(y_test_inv).flatten() & ~np.isnan(y_pred_inv).flatten()
    y_test_clean = y_test_inv[mask]
    y_pred_clean = y_pred_inv[mask]
    rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
    mae = mean_absolute_error(y_test_clean, y_pred_clean)
    print(f"✅ {name} RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    return y_test_clean, y_pred_clean

# LSTM 모델
def train_lstm():
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=0)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, scaler, "LSTM")

# GRU 모델
def train_gru():
    model = Sequential([
        GRU(64, activation='relu', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=0)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, scaler, "GRU")

# CNN 모델
def train_cnn():
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=0)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, scaler, "CNN")

# Transformer 모델
def train_transformer():
    input_layer = Input(shape=(X_train.shape[1], 1))
    x = LayerNormalization()(input_layer)
    attn_output = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=0)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, scaler, "Transformer")

# 실행
if __name__ == "__main__":
    train_lstm()
    train_gru()
    train_cnn()
    train_transformer()
