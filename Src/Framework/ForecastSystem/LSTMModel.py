from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import numpy as np
import MetaTrader5 as mt5
import pandas as pd

def create_sequences(df, sequence_length=150, target_column="close"):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    # 数値データだけ抽出（非数値列を除去）
    data = df.select_dtypes(include=['float64', 'int64']).values

    # 学習に使えない初期行を一気に削除
    df = df.dropna().reset_index(drop=True)
    print(df.shape)

    # 正規化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 入力Xと出力yを作成
    X, y = [], []
    target_index = df.columns.get_loc(target_column)

    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length][target_index])

    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # 出力は1つ（終値）
    model.compile(optimizer='adam', loss='mse')
    return model