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
