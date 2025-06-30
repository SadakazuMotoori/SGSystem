import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from ta.trend import ADXIndicator, PSARIndicator

def show_trend_labels_in_period(df, start_date, end_date):
    """
    指定された期間におけるトレンドラベルの確認用関数。
    """
    df_period = df.loc[start_date:end_date, ["close", "Trend_Label"]]
    print(f"\n[Trend 判定ログ] {start_date} 〜 {end_date}")
    print(df_period)
    
def apply_trend_labels(df, period=90, slope_threshold=0.05, adx_threshold=25, verbose=False):
    labels = [None] * len(df)

    for i in range(period, len(df)):
        sub_df = df.iloc[:i+1]  # 過去i本まで（t=iが現在）
        result = PhaseA_Filter(
            sub_df,
            period=period,
            slope_threshold=slope_threshold,
            adx_threshold=adx_threshold,
            verbose=verbose
        )
        labels[i] = result

    df["Trend_Label"] = labels
    return df

def PhaseA_Filter(df, period=90, slope_threshold=0.01, adx_threshold=25, verbose=True):
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


def PhaseB_Trigger(predicted_close: list[float], df: pd.DataFrame) -> tuple[str, str]:
    try:
        if not predicted_close or len(predicted_close) < 2:
            return "NO-TRADE", "予測値不足"

        latest = df.iloc[-1]
        direction_score = sum(predicted_close[i] < predicted_close[i + 1] for i in range(len(predicted_close) - 1))
        bearish_score = sum(predicted_close[i] > predicted_close[i + 1] for i in range(len(predicted_close) - 1))

        # ▼ 予測が横ばいの場合はSMA傾斜で方向を判断
        if direction_score == 0 and bearish_score == 0:
            sma_today = df["SMA_20"].iloc[-1]
            sma_past = df["SMA_20"].iloc[-6]
            sma_slope = sma_today - sma_past

            if sma_slope > 0:
                return "BUY", f"予測横ばい → SMA傾斜BUY判断 (傾き={sma_slope:.5f})"
            elif sma_slope < 0:
                return "SELL", f"予測横ばい → SMA傾斜SELL判断 (傾き={sma_slope:.5f})"
            else:
                return "NO-TRADE", f"予測横ばいかつSMAフラット (傾き={sma_slope:.5f})"

        # ▼ 通常の方向性あり判定
        if direction_score > bearish_score:
            return "BUY", f"予測↑:{direction_score} > ↓:{bearish_score}"
        elif bearish_score > direction_score:
            return "SELL", f"予測↓:{bearish_score} > ↑:{direction_score}"
        else:
            return "NO-TRADE", f"予測拮抗 (↑:{direction_score} = ↓:{bearish_score})"

    except Exception as e:
        return "NO-TRADE", f"PhaseB エラー: {str(e)}"
