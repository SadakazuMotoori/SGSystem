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
        "SMA_50", "ATR_14", "delta_close"
    ]
    feature_columns = [f for f in features if f != target_column]

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
    model.add(LSTM(units=64, return_sequences=False, input_shape=input_shape))
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

def train_and_predict_lstm(df, show_plt=False, evaluate=True):
    print("[INFO] 欠損値除去中...")
    df = df.dropna().copy()
    print(df.isnull().sum())

    print("[INFO] モデル構築と学習開始")
    X, y, target_scaler, feature_scaler = create_sequences(df)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

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

    val_pred = model.predict(X_val)

    if evaluate:
        y_val_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        val_pred_actual = target_scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
        evaluate_model(y_val_actual, val_pred_actual, show_plt)

    predictions = []
    future_df = df.copy()

    for i in range(5):
        X, y, target_scaler, feature_scaler = create_sequences(future_df)
        model = build_lstm_model((X.shape[1], X.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X[:-1], y[:-1],
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        next_input = X[-1].reshape(1, X.shape[1], X.shape[2])
        next_scaled = model.predict(next_input)

        predicted_close = target_scaler.inverse_transform(next_scaled)[0][0]
        predictions.append(predicted_close)

        new_row = future_df.iloc[-1].copy()
        new_row['close'] = predicted_close
        for col in future_df.columns:
            if col != 'close':
                new_row[col] = future_df.iloc[-1][col]
        future_df.loc[future_df.index[-1] + pd.Timedelta(days=1)] = new_row

    print("[PREDICTED] 5日間の予測終値:")
    for i, pred in enumerate(predictions, 1):
        print(f" Day {i}: {pred:.3f}")

    plot_days = 30
    past_dates = df.index[-plot_days:]
    past_values = df["close"].iloc[-plot_days:].tolist()

    last_date = df.index[-1].date()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]

    if show_plt:
        plt.figure(figsize=(12, 6))
        plt.plot(past_dates, past_values, label="Actual Close", color="blue", marker="o")
        plt.plot(future_dates, predictions, label="Predicted Close", color="red", linestyle="--", marker="x")
        plt.axvline(past_dates[-1], color="gray", linestyle=":", label="Prediction Start")
        plt.title("USD/JPY Close Price: Past 30 Days + 5-Day Forecast")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return predictions

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
