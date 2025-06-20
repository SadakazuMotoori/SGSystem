import os
import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
import ta
from datetime import datetime, timedelta
from Framework.ForecastSystem.LSTMModel import train_and_predict_lstm

symbol = "USDJPY"  # 利用する通貨ペア


def MTManager_Initialize():
    """MT5へ接続を試み、成功・失敗を表示"""
    print("[INFO] MTManager Initialize 開始")

    # 環境変数からログイン情報取得
    loginID = int(os.getenv('MT_LOGIN_ID'))
    loginPass = os.getenv('MT_LOGIN_PASS')

    # MT5初期化とログイン
    if not mt5.initialize(login=loginID, server="OANDA-Japan MT5 Live", password=loginPass):
        print("[ERROR] MT5接続失敗：", mt5.last_error())
        return False

    print("[INFO] MT5接続成功")
    return True


def MTManager_UpdateIndicators(days_back=600):
    """
    テクニカル指標の更新、チャート描画、LSTM学習実行までを一括で行う。
    :param days_back: 取得する日足データの過去日数（営業日ベース）
    :return: 加工済みDataFrame
    """
    print("[INFO] インジケータ更新と学習開始")

    # 取得開始日時（過去days_back営業日分）
    from_date = datetime.now() - timedelta(days=days_back)

    # MT5からローソク足データを取得
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_D1, from_date, days_back)
    if rates is None or len(rates) == 0:
        print("[ERROR] データ取得失敗")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    # ===== テクニカル指標追加 =====
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    macd = ta.trend.MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()

    df["Support"] = df["low"].rolling(window=10).min()
    df["Resistance"] = df["high"].rolling(window=10).max()

    # ===== チャート描画（任意） =====
    apds = [
        mpf.make_addplot(df["Support"], panel=0, color='green', linestyle='--', width=1),
        mpf.make_addplot(df["Resistance"], panel=0, color='red', linestyle='--', width=1),
        mpf.make_addplot(df["RSI_14"], panel=1, color='purple', ylabel='RSI'),
        mpf.make_addplot([30]*len(df), panel=1, color='gray', linestyle='--'),
        mpf.make_addplot([70]*len(df), panel=1, color='gray', linestyle='--'),
        mpf.make_addplot(df["MACD"], panel=2, color='blue', ylabel='MACD'),
        mpf.make_addplot(df["MACD_signal"], panel=2, color='orange'),
        mpf.make_addplot(df["MACD_diff"], panel=2, type='bar', color='dimgray', alpha=0.5)
    ]

    mpf.plot(df,
             type='candle',
             style='charles',
             mav=(5, 25, 75),
             volume=True,
             addplot=apds,
             panel_ratios=(4, 1, 1),
             title='USD/JPY - MA + RSI + MACD',
             ylabel='Price',
             ylabel_lower='Volume',
             figsize=(14, 10))

    # ===== LSTM実行（学習＋評価＋予測）=====
    train_and_predict_lstm(df)

    return df
