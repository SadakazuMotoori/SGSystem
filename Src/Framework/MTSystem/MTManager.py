
# ===================================================
# MTManager.py
# - MetaTrader5から為替データを取得・加工する中核モジュール
# - LSTMモデルに渡すためのインジケータ追加・可視化も担う
# ===================================================

import  os
import  MetaTrader5                         as mt5
import  pandas                              as pd
import  mplfinance                          as mpf
import  matplotlib.pyplot                   as plt
import  ta
from    ta.volatility                       import AverageTrueRange
from Framework.ForecastSystem.SignalEngine  import apply_trend_labels
from Framework.ForecastSystem.LSTMModel     import train_and_predict_lstm


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
def draw_chart_with_trend_labels(df):
    # 追加指標
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

    # チャート描画（Figure取得）
    fig, axes = mpf.plot(df,
                         type='candle',
                         style='charles',
                         mav=(5, 25, 75),
                         volume=True,
                         addplot=apds,
                         panel_ratios=(4, 1, 1),
                         title='USD/JPY - MA + RSI + MACD + Trend',
                         ylabel='Price',
                         ylabel_lower='Volume',
                         figsize=(14, 10),
                         returnfig=True)

    # ローソク足パネル
    ax_price = axes[0]

    # y軸調整用オフセット
    offset = (df["high"].max() - df["low"].min()) * 0.01

    # ===============================
    # トレンドラベルを矢印で描画する
    # ===============================
    for i in range(len(df)):
        label = df["Trend_Label"].iloc[i]

        if label == "uptrend":
            price = df["low"].iloc[i] * 0.995
            ax_price.scatter([i], [price], marker='^', color='green', s=80, zorder=5)
#            print(f"[PLOT] {df.index[i].date()} ↑ {price:.2f}")
        elif label == "downtrend":
            price = df["high"].iloc[i] * 1.005
            ax_price.scatter([i], [price], marker='v', color='red', s=80, zorder=5)
#            print(f"[PLOT] {df.index[i].date()} ↓ {price:.2f}")

    # y軸の表示範囲を調整
    ax_price.set_ylim(df["low"].min() - 3 * offset, df["high"].max() + 3 * offset)

    # レイアウト調整
    plt.tight_layout()
    plt.show()

def MTManager_UpdateIndicators(days_back=600):
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
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["SMA_50"]  = df["close"].rolling(window=50).mean()

    # 既存の指標計算（RSI, MACDなど）に加えて
    atr_indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["ATR_14"] = atr_indicator.average_true_range()

    # ADX + DI系を追加（PhaseA_Filter用）
    adx_calc = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["ADX_14"] = adx_calc.adx()
    df["+DI"] = adx_calc.adx_pos()
    df["-DI"] = adx_calc.adx_neg()

    # PSARを追加（PhaseA_Filter用）
    psar_calc = ta.trend.PSARIndicator(high=df["high"], low=df["low"], close=df["close"])
    df["PSAR"] = psar_calc.psar()

    # 🔽 追加（変化率指標）
    df["delta_close"] = df["close"].diff().fillna(0)

    # ===================================================
    # チャート描画用トレンドラベルを追記
    # ===================================================
    df = apply_trend_labels(df, period=60, slope_threshold=0.005, adx_threshold=20, verbose=False)

    # ===================================================
    # LSTMモデル実行（予測＆チャート表示）
    # ===================================================
    #train_and_predict_lstm(df,True)

    # ===================================================
    # 前日のトレンドラベルを確認
    # ===================================================
    trend_signal = None
    if len(df) >= 2:
        trend_label = df["Trend_Label"].iloc[-2]  # 最新の一本前（t-1）
        if trend_label in ["uptrend", "downtrend"]:
            trend_signal = trend_label
            print(f"[SIGNAL] 前日のシグナル：{trend_label}")
        else:
            print("[SIGNAL] 前日はノーシグナル")

    return df, trend_signal