import numpy                as np
import  MetaTrader5         as mt5
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
def LSTMModel_PredictLSTM(df, timeFrame = mt5.TIMEFRAME_D1, show_plot = False):
    print("[INFO] LSTM Phase開始")

    # LONG(日足)バージョン
    if timeFrame == mt5.TIMEFRAME_D1:
        _sequence_length    = 120
        _prediction_steps   = 5  # ← 5日後まで
    # SHORT(15分足)バージョン
    else:
        _sequence_length    = 48
        _prediction_steps   = 5  # ← 5日後まで

    FEATURES = [
        "close", "volume", "SMA_20", "SMA_50", "RSI_14",
        "MACD", "MACD_signal", "MACD_diff",
        "Support", "Resistance", "ATR_14",
        "ADX_14", "+DI", "-DI", "PSAR"
    ]

    df_feat = df[FEATURES].copy().dropna()
    df_target = df["close"].copy()

    # 特徴量とターゲットをスケーリング
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(df_feat)
    y_scaled = target_scaler.fit_transform(df_target.values.reshape(-1, 1))

    # シーケンスとターゲットを構築（マルチステップ）
    X, y = [], []
    for i in range(_sequence_length, len(X_scaled) - _prediction_steps):
        X.append(X_scaled[i-_sequence_length:i])
        y.append(y_scaled[i:i+_prediction_steps].flatten())

    X, y = np.array(X), np.array(y)
    print(f"[INFO] 学習データ: {X.shape}, 正解ラベル: {y.shape}")

    # モデル構築
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(_prediction_steps))  # 出力5個

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

    # オプションでカーブ表示
    if show_plot:
        plt.figure(figsize=(12, 5))
        plt.plot(y_true[:, 0], label="True")
        plt.plot(y_pred[:, 0], label="Predicted Day+1")
        plt.title("LSTM Forecast Day+1 vs Actual")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 最新シーケンスから未来5日間を予測
    latest_sequence = X_scaled[-_sequence_length:]
    latest_sequence = np.expand_dims(latest_sequence, axis=0)
    future_pred_scaled = model.predict(latest_sequence)[0]
    future_pred = target_scaler.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()

    print("[予測] 5日先までの終値:", [f"{p:.2f}" for p in future_pred])

    return future_pred.tolist(), df
