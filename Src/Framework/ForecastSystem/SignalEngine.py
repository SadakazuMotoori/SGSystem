import pandas as pd

def PhaseA_Filter(df: pd.DataFrame) -> bool:
    """
    Phase-A: フィルタ条件の判定
    条件①：SMA50 の傾き → 短期的な上昇トレンドがあるか
    条件②：ATRの水準 → ボラティリティが十分か

    Returns:
        bool: True なら通過、False ならフィルタ落ち
    """
    try:
        sma50_today = df.iloc[-1]["SMA_50"]
        sma50_yesterday = df.iloc[-2]["SMA_50"]
        atr = df.iloc[-1]["ATR_14"]

        # 短期上昇トレンドの判定
        trend_up = sma50_today > sma50_yesterday

        # ボラティリティの閾値（今後調整可）
        atr_threshold = 0.5
        vol_active = atr > atr_threshold

        # --- Debug 出力（必要に応じて有効化） ---
        print(f"[DEBUG] SMA_50: {sma50_yesterday:.4f} → {sma50_today:.4f}, ATR: {atr:.4f}")
        print(f"[DEBUG] trend_up={trend_up}, vol_active={vol_active}")

        return trend_up and vol_active

    except Exception as e:
        print(f"[PhaseA_Filter] エラー: {e}")
        return False


def PhaseB_Trigger(predicted_close: list[float], df: pd.DataFrame) -> str:
    """
    Phase-B: 予測とオシレータを使った仕掛け判定（BUY/SELL/NO-TRADE）

    Returns:
        str: "BUY", "SELL", or "NO-TRADE"
    """
    try:
        if not predicted_close or len(predicted_close) < 2:
            print("[PhaseB_Trigger] 不正な予測入力")
            return "NO-TRADE"

        latest = df.iloc[-1]

        # --- ① 予測トレンド方向のスコア ---
        direction_score = sum(
            predicted_close[i] < predicted_close[i + 1]
            for i in range(len(predicted_close) - 1)
        )
        bullish_pred = direction_score >= 3
        bearish_pred = direction_score <= 1

        # --- ② RSI・MACD条件（緩和版） ---
        rsi = latest.get("RSI_14", None)
        macd = latest.get("MACD", None)
        macd_signal = latest.get("MACD_signal", None)

        if rsi is None or macd is None or macd_signal is None:
            print("[PhaseB_Trigger] インジケータ欠損")
            return "NO-TRADE"

        # 仮のテスト用：極端に緩く
        bullish_osc = rsi < 80 and macd > macd_signal
        bearish_osc = rsi > 20 and macd < macd_signal

        # --- Debug 出力（必要に応じて有効化） ---
        print(f"[DEBUG] RSI={rsi}, MACD={macd}, Signal={macd_signal}")
        print(f"[DEBUG] bullish_pred={bullish_pred}, bullish_osc={bullish_osc}")
        print(f"[DEBUG] bearish_pred={bearish_pred}, bearish_osc={bearish_osc}")

        if bullish_pred and bullish_osc:
            return "BUY"
        elif bearish_pred and bearish_osc:
            return "SELL"
        else:
            return "NO-TRADE"

    except Exception as e:
        print(f"[PhaseB_Trigger] エラー: {e}")
        return "NO-TRADE"
