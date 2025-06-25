
# ===================================================
# MTManager.py
# - MetaTrader5から為替データを取得・加工する中核モジュール
# - LSTMモデルに渡すためのインジケータ追加・可視化も担う
# ===================================================

import  os
import  MetaTrader5                     as mt5
import  pandas                          as pd
import  mplfinance                      as mpf
import  ta
from    ta.volatility                   import AverageTrueRange
from datetime                           import datetime, timedelta
from Framework.ForecastSystem.LSTMModel import train_and_predict_lstm

# ---------------------------------------------------
# 使用する通貨ペア（MT5に接続して有効である必要がある）
# ---------------------------------------------------
symbol = "USDJPY"

# ===================================================
# MT5初期化＆ログイン
# - 環境変数からIDとパスを読み込み、OANDA MT5サーバへ接続
# ===================================================
def MTManager_Initialize():
    print("[INFO] MTManager Initialize 開始")

    loginID = int(os.getenv('MT_LOGIN_ID'))
    loginPass = os.getenv('MT_LOGIN_PASS')

    if not mt5.initialize(login=loginID, server="OANDA-Japan MT5 Live", password=loginPass):
        print("[ERROR] MT5接続失敗：", mt5.last_error())
        return False

    print("[INFO] MT5接続成功")
    return True

# ===================================================
# 日足データ取得＆インジケータ追加
# - 最新日から過去へ指定数分取得（営業日ベース）
# - RSI・MACD・サポレジを計算し、チャートを表示
# ===================================================
def MTManager_UpdateIndicators(days_back=600, show_prot = True):
    print("[INFO] インジケータ更新と学習開始")

    # MT5からローソク足データを取得（最新からdays_back件分）
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, days_back)
    if rates is None or len(rates) == 0:
        print("[ERROR] データ取得失敗")
        return None

    # データフレーム化・インデックス変換
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    # ===================================================
    # テクニカル指標の計算
    # ===================================================
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    macd = ta.trend.MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()

    df["Support"] = df["low"].rolling(window=10).min()
    df["Resistance"] = df["high"].rolling(window=10).max()

    # SMAを追加（Phase-Aフィルタで必要）
    df["SMA_50"]  = df["close"].rolling(window=50).mean()
    # モデル精度への影響が大きいため除外
    #df["SMA_200"] = df["close"].rolling(window=200).mean()

    # 既存の指標計算（RSI, MACDなど）に加えて
    atr_indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["ATR_14"] = atr_indicator.average_true_range()

    # 🔽 追加（変化率指標）
    df["delta_close"] = df["close"].diff().fillna(0)

    # ===================================================
    # チャート描画（ローソク足＋インジケータ）
    # ===================================================
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

    if(show_prot):
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

    # ===================================================
    # LSTMモデル実行（予測＆チャート表示）
    # ===================================================
    train_and_predict_lstm(df)

    return df

# ==============================
# Position State Management for Backtest
# ==============================
_position_active    = False
_position_end_index = -1  # インデックスベースでの保有期間終端

def ResetPositionState():
    global _position_active, _position_end_index
    _position_active = False
    _position_end_index = -1

def IsPositionActive(current_index=None):
    global _position_active, _position_end_index
    if current_index is not None:
        return _position_active and current_index <= _position_end_index
    return _position_active

def SetPositionActive(period_length, current_index):
    global _position_active, _position_end_index
    _position_active = True
    _position_end_index = current_index + period_length

def ClosePosition():
    global _position_active
    _position_active = False