import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt

from sklearn.preprocessing  import MinMaxScaler
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score
from keras.models           import Sequential
from keras.layers           import LSTM, Dense, Dropout
from keras.callbacks        import EarlyStopping

def create_sequences(df, sequence_length=150, target_column="close"):
    features = [
        "open", "high", "low", "close", "volume", "spread", "real_volume",
        "RSI_14", "MACD", "MACD_signal", "MACD_diff",
        "Support", "Resistance",
        "SMA_50", "ATR_14", "delta_close",
        "ADX_14", "+DI", "-DI", "PSAR"
    ]
    feature_columns = [f for f in features if f != target_column]

    # 欠損がある行は除去（とくにRSI・MACD系に注意）
    df = df.dropna(subset=features).copy()

    target_scaler = MinMaxScaler()
    feature_scaler = MinMaxScaler()

    y_scaled = target_scaler.fit_transform(df[[target_column]])
    X_scaled = feature_scaler.fit_transform(df[feature_columns])

    scaled_df = np.concatenate([y_scaled, X_scaled], axis=1)

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(scaled_df[i-sequence_length:i])
        y.append(scaled_df[i][0])

    return np.array(X), np.array(y), target_scaler, feature_scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred, show_plt):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n[Model Evaluation]")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")

    if(show_plt):
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.title("Actual vs Predicted Close Price (Validation)")
        plt.xlabel("Time")
        plt.ylabel("Close Price (JPY)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ===================================================
# LSTMモデルの学習・予測
# - 入力: 特徴量付きDataFrame（df）
# - 出力: 翌日の終値予測値（1ステップ）
# ===================================================
def train_and_predict_lstm(df, show_plot=False):
    print("[INFO] LSTM Phase開始")

    # ===================================================
    # 特徴量選定（過去60日分を使って翌日の終値を予測）
    # ===================================================
    sequence_length = 120
    FEATURES = [
        "close", "volume", "SMA_20", "SMA_50", "RSI_14",
        "MACD", "MACD_signal", "MACD_diff",
        "Support", "Resistance", "ATR_14",
        "ADX_14", "+DI", "-DI", "PSAR"
    ]

    df_feat = df[FEATURES].copy().dropna()
    df_target = df["close"].shift(-1).dropna()

    # 同期処理
    min_len = min(len(df_feat), len(df_target))
    df_feat, df_target = df_feat[:min_len], df_target[:min_len]

    # 特徴量とターゲットを個別にスケーリング
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(df_feat)
    y_scaled = target_scaler.fit_transform(df_target.values.reshape(-1, 1))

    # シーケンス化（60ステップ）
    X, y = [], []
    for i in range(sequence_length, len(X_scaled)):
        X.append(X_scaled[i-sequence_length:i])
        y.append(y_scaled[i])  # y もスケーリング後を使用

    X, y = np.array(X), np.array(y)
    print(f"[INFO] 学習データ: {X.shape}, 正解ラベル: {y.shape}")

    # ===================================================
    # モデル構築（2層LSTM + Dropout）
    # ===================================================
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=30, batch_size=32, verbose=0)

    # ===================================================
    # 予測と評価（逆スケーリングしてから評価）
    # ===================================================
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

    # ===================================================
    # 可視化（希望時のみ）
    # ===================================================
    if show_plot:
        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Predicted")
        plt.title("LSTM Forecast vs Actual")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ===================================================
    # 直近の予測を返す（[-1]）
    # ===================================================
    latest_sequence = X_scaled[-sequence_length:]
    latest_sequence = np.expand_dims(latest_sequence, axis=0)
    predicted_scaled = model.predict(latest_sequence)[0][0]
    predicted_price = target_scaler.inverse_transform([[predicted_scaled]])[0][0]
    print(f"[予測] 翌日の終値: {predicted_price:.3f}")

    return predicted_price

def generate_predicted_series(df, sequence_length=90, target_column="close"):
    print("[INFO] 逐次予測シリーズ生成を開始...")
    df = df.dropna().copy()

    features = [
        "open", "high", "low", "close", "volume", "spread", "real_volume",
        "RSI_14", "MACD", "MACD_signal", "MACD_diff",
        "Support", "Resistance",
        "SMA_50", "ATR_14", "delta_close"
    ]
    feature_columns = [f for f in features if f != target_column]

    target_scaler = MinMaxScaler()
    feature_scaler = MinMaxScaler()

    y_scaled = target_scaler.fit_transform(df[[target_column]])
    X_scaled = feature_scaler.fit_transform(df[feature_columns])
    scaled_df = np.concatenate([y_scaled, X_scaled], axis=1)

    def build_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=False, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    predicted_series = [None] * len(df)

    for i in range(sequence_length, len(df)):
        X_seq = scaled_df[i-sequence_length:i].reshape(1, sequence_length, -1)
        y_target = scaled_df[i][0]

        X_train = scaled_df[i-sequence_length:i].reshape(1, sequence_length, -1)
        y_train = np.array([y_target])

        model = build_lstm_model((sequence_length, X_train.shape[2]))
        model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=0)

        y_pred_scaled = model.predict(X_seq)
        y_pred_actual = target_scaler.inverse_transform(y_pred_scaled)[0][0]

        predicted_series[i] = y_pred_actual

    for j in range(sequence_length):
        predicted_series[j] = predicted_series[sequence_length] if predicted_series[sequence_length] is not None else None

    print("[INFO] 逐次予測完了。")
    return predicted_series
