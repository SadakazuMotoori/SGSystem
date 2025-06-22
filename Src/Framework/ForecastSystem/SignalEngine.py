# SignalEngine.py
import pandas as pd

def PhaseA_Filter(df: pd.DataFrame) -> bool:
    """
    Phase-A: フィルタ条件の判定
    条件①：SMA50 の傾き or 前日比から短期トレンドを判断
    条件②：ATRの水準からボラティリティを判断

    Returns:
        bool: True なら通過、False ならフィルタ落ち
    """
    try:
        # 直近2日の SMA50 を比較
        sma50_today = df.iloc[-1]["SMA_50"]
        sma50_yesterday = df.iloc[-2]["SMA_50"]
        atr = df.iloc[-1]["ATR_14"]

        # 条件①：SMA50が上昇傾向
        trend_up = sma50_today > sma50_yesterday

        # 条件②：ATRが一定以上
        atr_threshold = 0.5  # この値は今後調整対象
        vol_active = atr > atr_threshold

        return trend_up and vol_active

    except Exception as e:
        print(f"[PhaseA_Filter] エラー: {e}")
        return False
    
def PhaseB_Trigger(predicted_close: list[float], df: pd.DataFrame) -> str:
    """
    Phase-B: 予測とオシレータを使った仕掛け判定（条件緩和版）
    Returns:
        "BUY", "SELL", or "NO-TRADE"
    """
    try:
        latest = df.iloc[-1]

        # --- ① 予測の方向性をチェック ---
        direction_score = sum([predicted_close[i] < predicted_close[i + 1] for i in range(len(predicted_close) - 1)])
        bullish_pred = direction_score >= 3
        bearish_pred = direction_score <= 1

        # --- ② RSIとMACD条件（緩和） ---
        rsi = latest["RSI_14"]
        macd = latest["MACD"]
        macd_signal = latest["MACD_signal"]

        # 緩和条件：
        # - ロング: RSIが40未満（以前は30）、MACD > Signal
        # - ショート: RSIが60超（以前は70）、MACD < Signal
        bullish_osc = rsi < 40 and macd > macd_signal
        bearish_osc = rsi > 60 and macd < macd_signal

        if bullish_pred and bullish_osc:
            return "BUY"
        elif bearish_pred and bearish_osc:
            return "SELL"
        else:
            return "NO-TRADE"

    except Exception as e:
        print(f"[PhaseB_Trigger] エラー: {e}")
        return "NO-TRADE"

