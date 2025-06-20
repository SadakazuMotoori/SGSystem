import os
import MetaTrader5 as mt5
import pandas as pd
import ta
from datetime import datetime

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
    rate        = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, 1)

    # DataFrame化
    df          = pd.DataFrame(rate)
    df['time']  = pd.to_datetime(df['time'], unit='s')

    # --- 移動平均線 ---
    df["MA_5"]          = df["close"].rolling(window=5).mean()
    df["MA_25"]         = df["close"].rolling(window=25).mean()
    df["MA_75"]         = df["close"].rolling(window=75).mean()

    # --- RSI ---
    rsi                 = ta.momentum.RSIIndicator(close=df["close"], window=14)
    df["RSI_14"]        = rsi.rsi()

    # --- MACD ---
    macd                = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]          = macd.macd()
    df["MACD_signal"]   = macd.macd_signal()
    df["MACD_diff"]     = macd.macd_diff()

    # --- サポート／レジスタンス（簡易） ---
    df["Support"]       = df["low"].rolling(window=10).min()
    df["Resistance"]    = df["high"].rolling(window=10).max()

    print(df.tail())