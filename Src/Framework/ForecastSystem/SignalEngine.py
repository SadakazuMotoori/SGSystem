import pandas as pd

def PhaseA_Filter(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        if len(df) < 2:
            return False, "データ不足 (<2行)"

        sma50_today = df.iloc[-1]["SMA_50"]
        sma50_yesterday = df.iloc[-2]["SMA_50"]
        atr = df.iloc[-1]["ATR_14"]

        trend_up = sma50_today > sma50_yesterday
        atr_threshold = 0.5
        vol_active = atr > atr_threshold

        if not trend_up and not vol_active:
            return False, "SMA下降 & ATR不足"
        elif not trend_up:
            return False, "SMA下降"
        elif not vol_active:
            return False, "ATR不足"
        else:
            return True, "通過"

    except Exception as e:
        return False, f"PhaseA エラー: {str(e)}"


def PhaseB_Trigger(predicted_close: list[float], df: pd.DataFrame) -> tuple[str, str]:
    try:
        if not predicted_close or len(predicted_close) < 2:
            return "NO-TRADE", "予測値不足"

        latest = df.iloc[-1]
        direction_score = sum(predicted_close[i] < predicted_close[i + 1] for i in range(len(predicted_close) - 1))
        bearish_score = sum(predicted_close[i] > predicted_close[i + 1] for i in range(len(predicted_close) - 1))

        bullish_pred = direction_score >= 3
        bearish_pred = bearish_score >= 3

        rsi_now = latest.get("RSI_14", None)
        rsi_prev = df.iloc[-2].get("RSI_14", None)
        macd_now = latest.get("MACD", None)
        macd_prev = df.iloc[-2].get("MACD", None)
        macd_sig_now = latest.get("MACD_signal", None)
        macd_sig_prev = df.iloc[-2].get("MACD_signal", None)

        if None in [rsi_now, rsi_prev, macd_now, macd_prev, macd_sig_now, macd_sig_prev]:
            return "NO-TRADE", "インジケータ欠損"

        rsi_avg = (rsi_now + rsi_prev) / 2
        bullish_rsi = rsi_avg < 35
        bearish_rsi = rsi_avg > 65
        bullish_macd = macd_now > macd_sig_now and macd_prev > macd_sig_prev
        bearish_macd = macd_now < macd_sig_now and macd_prev < macd_sig_prev

        bullish_osc = bullish_rsi or bullish_macd
        bearish_osc = bearish_rsi or bearish_macd

        reason_log = f"[DEBUG] DIR={direction_score}, RSI_avg={rsi_avg:.2f}, MACD_now={macd_now:.4f}, Signal_now={macd_sig_now:.4f}"

        if bullish_pred and bullish_osc:
            return "BUY", f"予測↑:{direction_score} & RSI or MACD 強気 | {reason_log}"
        elif bearish_pred and bearish_osc:
            return "SELL", f"予測↓:{bearish_score} & RSI or MACD 弱気 | {reason_log}"
        else:
            reasons = []
            if not bullish_pred and not bearish_pred:
                reasons.append("方向性弱い")
            if not bullish_osc and not bearish_osc:
                reasons.append("オシレータ不一致")
            return "NO-TRADE", " / ".join(reasons) + " | " + reason_log

    except Exception as e:
        return "NO-TRADE", f"PhaseB エラー: {str(e)}"
