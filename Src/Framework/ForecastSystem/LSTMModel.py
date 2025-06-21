import numpy                as np
import matplotlib.pyplot    as plt

from sklearn.preprocessing  import MinMaxScaler
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score
from keras.models           import Sequential
from keras.layers           import LSTM, Dense, Dropout
from keras.callbacks        import EarlyStopping

# ======================
# シーケンス生成関数
# ======================
def create_sequences(df, sequence_length=150, target_column="close"):
    feature_columns = df.columns.tolist()
    feature_columns.remove(target_column)
    feature_columns = [target_column] + feature_columns  # targetを先頭に

    # スケーリング
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i][0])  # close列（先頭）のみ

    return np.array(X), np.array(y), scaler

# ======================
# モデル構築関数
# ======================
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ======================
# 評価指標と可視化関数
# ======================
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n[Model Evaluation]")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Actual vs Predicted (Validation)")
    plt.xlabel("Time")
    plt.ylabel("Scaled Close")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ======================
# 学習＆予測主関数
# ======================
def train_and_predict_lstm(df, evaluate=True):
    print("[INFO] 欠損値除去中...")
    df = df.dropna().copy()
    print(df.isnull().sum())

    print("[INFO] モデル構築と学習開始")
    
    # 念のため create_sequences 前にも再度 dropna する
    df = df.dropna().copy()

    X, y, scaler = create_sequences(df)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # 学習/検証データ分割
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm_model((X.shape[1], X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # 検証データの予測
    val_pred = model.predict(X_val)

    if evaluate:
        evaluate_model(y_val, val_pred)

    # 翌日の予測（直近シーケンス）
    next_input = X[-1].reshape(1, X.shape[1], X.shape[2])
    next_scaled = model.predict(next_input)
    predicted_close = scaler.inverse_transform(
        np.concatenate([next_scaled, np.zeros((1, X.shape[2] - 1))], axis=1))[0][0]

    print(f"[PREDICTED] 翌日の予測終値: {predicted_close:.3f}")
    return predicted_close