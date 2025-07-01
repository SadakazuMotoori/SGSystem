import numpy                as np

from sklearn.linear_model   import LinearRegression
from ta.trend               import ADXIndicator, PSARIndicator

def __PhaseA_Filter(df, period=90, slope_threshold=0.01, adx_threshold=25, verbose=True):
    """
    t-1時点を基点に、トレンド方向を 'uptrend', 'downtrend', 'no_trend' のいずれかで判定。
    判定条件：
    - 線形回帰による終値傾き
    - ADX > adx_threshold
    - SMA20 > SMA50（上昇） / SMA20 < SMA50（下降）
    - PSARが価格の下（上昇） / 上（下降）
    """

    if len(df) < period + 1:
        if verbose:
            print("[WARN] データ不足：{}本必要（現在{}本）".format(period+1, len(df)))
        return "no_trend"

    # 対象データ：t-N〜t-1（直近は除外）
    sub_df = df.iloc[-period-1:-1]
    y = sub_df["close"].values.reshape(-1, 1)
    X = np.arange(period).reshape(-1, 1)
    slope = LinearRegression().fit(X, y).coef_[0][0]

    # ADX
    if "ADX_14" not in df.columns:
        adx_calc = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["ADX_14"] = adx_calc.adx()
        df["+DI"] = adx_calc.adx_pos()
        df["-DI"] = adx_calc.adx_neg()

    adx_val = df["ADX_14"].iloc[-2]
    plus_di = df["+DI"].iloc[-2]
    minus_di = df["-DI"].iloc[-2]

    # SMA
    sma20 = df["SMA_20"].iloc[-2]
    sma50 = df["SMA_50"].iloc[-2]

    # PSAR
    if "PSAR" not in df.columns:
        psar_calc = PSARIndicator(high=df["high"], low=df["low"], close=df["close"])
        df["PSAR"] = psar_calc.psar()

    price_t1 = df["close"].iloc[-2]
    psar_val = df["PSAR"].iloc[-2]

    # --- ログ出力 ---
    if verbose:
        print("=== PhaseA_Filter 判定ログ ===")
        print(f"[期間]         過去{period}本（t-1基準）")
        print(f"[SLOPE]        {slope:.5f}（閾値 ±{slope_threshold}）")
        print(f"[ADX]          {adx_val:.2f}（閾値 {adx_threshold}）")
        print(f"[+DI/-DI]      +DI={plus_di:.2f}, -DI={minus_di:.2f}")
        print(f"[SMA]          SMA20={sma20:.2f}, SMA50={sma50:.2f}")
        print(f"[PSAR]         PSAR={psar_val:.2f}, close(t-1)={price_t1:.2f}")
        print("-------------------------------")

    # --- 判定ロジック ---
    if (
        slope > slope_threshold and
        adx_val > adx_threshold and
        sma20 > sma50 and
        plus_di > minus_di and
        psar_val < price_t1
    ):
        result = "uptrend"
    elif (
        slope < -slope_threshold and
        adx_val > adx_threshold and
        sma20 < sma50 and
        minus_di > plus_di and
        psar_val > price_t1
    ):
        result = "downtrend"
    else:
        result = "no_trend"

    if verbose:
        print(f"[RESULT]       ⇒ {result.upper()}")
        print("==============================\n")

    return result

def SignalEngine_PhaseA_Filter(df, period=90, slope_threshold=0.05, adx_threshold=25, verbose=False):
    labels = [None] * len(df)

    for i in range(period, len(df)):
        sub_df = df.iloc[:i+1]  # 過去i本まで（t=iが現在）
        result = __PhaseA_Filter( sub_df,
                                  period=period,
                                  slope_threshold=slope_threshold,
                                  adx_threshold=adx_threshold,
                                  verbose=verbose
                                )
        labels[i] = result

    df["Trend_Label"] = labels
    return df

def SignalEngine_PhaseB_Trigger(df):
    result = False

    # try:
    # except Exception as e:
        # return "NO-TRADE", f"PhaseB エラー: {str(e)}"
    
    return result