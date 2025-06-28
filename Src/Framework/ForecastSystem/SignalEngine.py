import pandas as pd

def PhaseA_Filter(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        if len(df) < 6:
            return False, "データ不足 (<6行)"

        sma_today = df["SMA_20"].iloc[-1]
        sma_past = df["SMA_20"].iloc[-6]
        atr_today = df["ATR_14"].iloc[-1]

        # 緩和された閾値設定
        atr_threshold_ratio = 0.3  # 従来の0.4より緩め
        fixed_threshold = 0.03     # 最小ライン（pips変換で約3pips相当）
        atr_threshold = atr_today * atr_threshold_ratio
        # 修正後（正しく最低限のフィルタとして機能させる）
        dynamic_threshold = max(fixed_threshold, atr_threshold)

        sma_diff = sma_today - sma_past

        if abs(sma_diff) > dynamic_threshold:
            return True, "通過"
        else:
            return False, f"PhaseA不成立: SMA変化={sma_diff:.5f}, ATR={atr_today:.5f}, 閾値={dynamic_threshold:.5f}"

    except Exception as e:
        return False, f"PhaseA エラー: {str(e)}"


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
