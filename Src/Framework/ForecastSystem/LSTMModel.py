import numpy                as np
import matplotlib.pyplot    as plt

from sklearn.preprocessing  import MinMaxScaler
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score
from keras.models           import Sequential
from keras.layers           import LSTM, Dense, Dropout

# ===================================================
# LSTMモデルの学習・予測
# - 入力: 特徴量付きDataFrame（df）
# - 出力: 翌日の終値予測値（1ステップ）と更新済みdf
# ===================================================
def LSTMModel_PredictLSTM(df, show_plot=False):
    print("[INFO] LSTM Phase開始")

    sequence_length = 120
    FEATURES = [
        "close", "volume", "SMA_20", "SMA_50", "RSI_14",
        "MACD", "MACD_signal", "MACD_diff",
        "Support", "Resistance", "ATR_14",
        "ADX_14", "+DI", "-DI", "PSAR"
    ]

    df_feat = df[FEATURES].copy().dropna()
    df_target = df["close"].shift(-1).dropna()

    min_len = min(len(df_feat), len(df_target))
    df_feat, df_target = df_feat[:min_len], df_target[:min_len]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(df_feat)
    y_scaled = target_scaler.fit_transform(df_target.values.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(X_scaled)):
        X.append(X_scaled[i-sequence_length:i])
        y.append(y_scaled[i])

    X, y = np.array(X), np.array(y)
    print(f"[INFO] 学習データ: {X.shape}, 正解ラベル: {y.shape}")

    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=30, batch_size=32, verbose=0)

    y_pred_scaled = model.predict(X)
    y_true = target_scaler.inverse_transform(y)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("[Model Evaluation]")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")

    if show_plot:
        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Predicted")
        plt.title("LSTM Forecast vs Actual")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    latest_sequence = X_scaled[-sequence_length:]
    latest_sequence = np.expand_dims(latest_sequence, axis=0)
    predicted_scaled = model.predict(latest_sequence)[0][0]
    predicted_price = target_scaler.inverse_transform([[predicted_scaled]])[0][0]
    print(f"[予測] 翌日の終値: {predicted_price:.3f}")

    return predicted_price, df
