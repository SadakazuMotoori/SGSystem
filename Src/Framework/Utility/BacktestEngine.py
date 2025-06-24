import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

def is_buy_signal(predicted_close, rsi, macd, macd_signal):
    if any(p is None or (isinstance(p, float) and np.isnan(p)) for p in predicted_close):
        return False
    """
    Phase-B ロジックに準拠したBUY条件の判定
    - RSIが70未満
    - MACD > Signal
    - 予測方向が上昇傾向（5ステップ中3回以上上昇）
    """
    direction_score = sum([predicted_close[i] < predicted_close[i + 1] for i in range(len(predicted_close) - 1)])
    bullish_pred = direction_score >= 3
    bullish_osc = rsi < 70 and macd > macd_signal
    return bullish_pred and bullish_osc

def is_sell_signal(predicted_close, rsi, macd, macd_signal):
    if any(p is None or (isinstance(p, float) and np.isnan(p)) for p in predicted_close):
        return False
    """
    Phase-B ロジックに準拠したSELL条件の判定
    - RSIが30超（売られすぎでない状態）
    - MACD < Signal
    - 予測方向が下落傾向（5ステップ中3回以上下降）
    """
    direction_score = sum([predicted_close[i] > predicted_close[i + 1] for i in range(len(predicted_close) - 1)])
    bearish_pred = direction_score >= 3
    bearish_osc = rsi > 30 and macd < macd_signal
    return bearish_pred and bearish_osc

def run_backtest(df, predicted_close, period_days=3, lookback=3):
    """
    柔軟なTP条件（直近x本での最高値/最安値更新）を使ったバックテスト
    """
    total_trades = 0
    win_trades = 0
    loss_trades = 0
    total_profit = 0.0
    max_drawdown = 0.0
    entry_logs = []

    df = df.copy().reset_index(drop=False)

    for i in range(len(df) - period_days):
        rsi = df.loc[i, "RSI_14"]
        macd = df.loc[i, "MACD"]
        macd_signal = df.loc[i, "MACD_signal"]
        close_entry = df.loc[i, "close"]

        local_pred = predicted_close[i:i+5]
        if len(local_pred) < 5:
            continue

        # === BUY ===
        if is_buy_signal(local_pred, rsi, macd, macd_signal):
            total_trades += 1
            entry_price = close_entry
            entry_index = i
            success = False

            for j in range(1, period_days + 1):
                future_price = df.loc[i + j, "close"]
                start_idx = max(0, i + j - lookback + 1)
                recent_high = df.loc[start_idx: i + j, "close"].max()

                drawdown = entry_price - future_price
                max_drawdown = max(max_drawdown, drawdown)

                if future_price >= recent_high:
                    profit = future_price - entry_price
                    total_profit += profit
                    win_trades += 1
                    success = True
                    entry_logs.append({
                        "date": df.loc[i + j, "time"],
                        "entry_price": future_price,
                        "type": "TP"
                    })
                    break

            if not success:
                profit = df.loc[i + period_days, "close"] - entry_price
                total_profit += profit
                loss_trades += 1

            entry_logs.append({
                "date": df.loc[entry_index, "time"],
                "entry_price": entry_price,
                "type": "BUY",
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "profit": profit,
                "success": success
            })

        # === SELL ===
        elif is_sell_signal(local_pred, rsi, macd, macd_signal):
            total_trades += 1
            entry_price = close_entry
            entry_index = i
            success = False

            for j in range(1, period_days + 1):
                future_price = df.loc[i + j, "close"]
                start_idx = max(0, i + j - lookback + 1)
                recent_low = df.loc[start_idx: i + j, "close"].min()

                drawdown = future_price - entry_price
                max_drawdown = max(max_drawdown, drawdown)

                if future_price <= recent_low:
                    profit = entry_price - future_price
                    total_profit += profit
                    win_trades += 1
                    success = True
                    entry_logs.append({
                        "date": df.loc[i + j, "time"],
                        "entry_price": future_price,
                        "type": "TP"
                    })
                    break

            if not success:
                profit = entry_price - df.loc[i + period_days, "close"]
                total_profit += profit
                loss_trades += 1

            entry_logs.append({
                "date": df.loc[entry_index, "time"],
                "entry_price": entry_price,
                "type": "SELL",
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "profit": profit,
                "success": success
            })

    result = {
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": (win_trades / total_trades) * 100 if total_trades > 0 else 0,
        "average_profit": (total_profit / total_trades) if total_trades > 0 else 0,
        "max_drawdown": max_drawdown,
        "entry_logs": entry_logs
    }

    if len(entry_logs) > 0:
        plot_entry_points_chart(df, entry_logs)

    return result

def plot_entry_points_chart(df, entry_logs):
    import pandas as pd
    import mplfinance as mpf
    import matplotlib.pyplot as plt

    # --- データフレームの準備 ---
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])       # time列をdatetime型に変換
    df.set_index("time", inplace=True)            # time列をインデックス化（mplfinance準拠）

    # --- OHLCV形式に整形 ---
    ohlc = df[["open", "high", "low", "close", "volume"]]  # チャート描画用データ

    # === BUYエントリーの処理 ===
    buys = [log for log in entry_logs if log["type"] == "BUY"]
    buy_df = pd.DataFrame({
        "date": pd.to_datetime([log["date"] for log in buys]),
        "price": [log["entry_price"] for log in buys]
    })
    buy_df = buy_df[buy_df["date"].isin(ohlc.index)]
    x_buys = [ohlc.index.get_loc(date) for date in buy_df["date"]]
    y_buys = buy_df["price"].values
    print(f"[DEBUG] BUY marker count after filtering: {len(buy_df)}")

    # === SELLエントリーの処理 ===
    sells = [log for log in entry_logs if log["type"] == "SELL"]
    sell_df = pd.DataFrame({
        "date": pd.to_datetime([log["date"] for log in sells]),
        "price": [log["entry_price"] for log in sells]
    })
    sell_df = sell_df[sell_df["date"].isin(ohlc.index)]
    x_sells = [ohlc.index.get_loc(date) for date in sell_df["date"]]
    y_sells = sell_df["price"].values
    print(f"[DEBUG] SELL marker count after filtering: {len(sell_df)}")

    # === TP（利確）マーカー処理 ===
    take_profits = [log for log in entry_logs if log["type"] == "TP"]
    tp_df = pd.DataFrame({
        "date": pd.to_datetime([log["date"] for log in take_profits]),
        "price": [log["entry_price"] for log in take_profits]
    })
    tp_df = tp_df[tp_df["date"].isin(ohlc.index)]
    x_tp = [ohlc.index.get_loc(date) for date in tp_df["date"]]
    y_tp = tp_df["price"].values
    print(f"[DEBUG] TP marker count after filtering: {len(tp_df)}")

    # === 整合性チェック ===
    if len(x_buys) != len(y_buys) or len(x_sells) != len(y_sells) or len(x_tp) != len(y_tp):
        print("[ERROR] XとYの長さ不一致：描画中止")
        return

    if len(x_buys) == 0 and len(x_sells) == 0 and len(x_tp) == 0:
        print("[INFO] BUY/SELL/TPエントリーが存在しません。チャート描画スキップ。")
        return

    # --- チャート描画 ---
    fig, axlist = mpf.plot(
        ohlc,
        type="candle",
        style="yahoo",
        title="Entry Points Chart",
        ylabel="Price",
        volume=True,
        returnfig=True,
        warn_too_much_data=len(ohlc)+1
    )

    ax_price = axlist[0]

    if len(x_buys) > 0:
        ax_price.scatter(x_buys, y_buys, marker='^', color='green', s=100, label='BUY')
    if len(x_sells) > 0:
        ax_price.scatter(x_sells, y_sells, marker='v', color='red', s=100, label='SELL')
    if len(x_tp) > 0:
        ax_price.scatter(x_tp, y_tp, marker='o', facecolors='none', edgecolors='blue', s=70, label='TP')

    ax_price.legend()
    plt.show()