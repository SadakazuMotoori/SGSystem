import os
import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
import ta
from datetime import datetime

from Framework.ForecastSystem.LSTMModel import create_sequences
from Framework.ForecastSystem.LSTMModel import build_lstm_model

"""
MetaTraderシステムとの連携準備
"""
# シンボル設定（ブローカーの仕様による。必要に応じて調整）
symbol = "USDJPY"

def MTManager_Initialize():
    print("MTManager Initialize")

    # MT5へ接続
    loginID     = int(os.getenv('MT_LOGIN_ID'))
    loginPass   = os.getenv('MT_LOGIN_PASS')
    if not mt5.initialize(login=loginID, server="OANDA-Japan MT5 Live",password=loginPass):
        print("接続失敗：", mt5.last_error())
        return False
    
    return True

"""
入力されたDataFrameに対して、各種テクニカル指標を追加する。
"""
def MTManager_UpadteIndicators():

    # 過去100日分のD1ローソク足取得
    rates       = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 200)

    # DataFrame化
    df          = pd.DataFrame(rates)
    df['time']  = pd.to_datetime(df['time'], unit='s')

    # インデックス設定（mplfinance形式に変換）
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    # --- RSI ---
    rsi = ta.momentum.RSIIndicator(close=df["close"], window=14)
    df["RSI_14"] = rsi.rsi()

    # --- MACD ---
    macd = ta.trend.MACD(close=df["close"])
    df["MACD"]          = macd.macd()
    df["MACD_signal"]   = macd.macd_signal()
    df["MACD_diff"]     = macd.macd_diff()

    # --- サポレジライン ---
    df["Support"]       = df["low"].rolling(window=10).min()
    df["Resistance"]    = df["high"].rolling(window=10).max()

    # --- サブチャート設定 ---
    apds = [
        # ✅ サポレジライン（パネル0 = メインチャート）
        mpf.make_addplot(df["Support"], panel=0, color='green', linestyle='--', width=1),
        mpf.make_addplot(df["Resistance"], panel=0, color='red', linestyle='--', width=1),

        # RSI線
        mpf.make_addplot(df["RSI_14"], panel=1, color='purple', ylabel='RSI'),
        # RSIガイドライン（30, 70）
        mpf.make_addplot([30]*len(df), panel=1, color='gray', linestyle='--'),
        mpf.make_addplot([70]*len(df), panel=1, color='gray', linestyle='--'),

        # MACD線・シグナル線・ヒストグラム
        mpf.make_addplot(df["MACD"], panel=2, color='blue', ylabel='MACD'),
        mpf.make_addplot(df["MACD_signal"], panel=2, color='orange'),
        mpf.make_addplot(df["MACD_diff"], panel=2, type='bar', color='dimgray', alpha=0.5)
    ]

    # --- チャート描画 ---
    mpf.plot(df,
             type='candle',
             style='charles',
             mav=(5, 25, 75),
             volume=True,
             addplot=apds,
             panel_ratios=(4, 1, 1),
             title='USD/JPY - MA + RSI + MACD (Final Layout)',
             ylabel='Price',
             ylabel_lower='Volume',
             figsize=(14, 10))

    X, y, scaler = create_sequences(df, sequence_length=90, target_column="close")
    
    import numpy as np
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("y NaN check:", np.isnan(y).sum())
    print("X NaN check:", np.isnan(X).sum())

    # LSTMモデルを構築
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))

    # 学習実行
    history = model.fit(
        X, y,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )

    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()