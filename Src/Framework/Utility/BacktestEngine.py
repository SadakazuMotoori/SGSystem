
import matplotlib.pyplot as plt

from Framework.MTSystem.MTManager import (
    IsPositionActive,
    SetPositionActive,
    ResetPositionState,
    ClosePosition
)

def is_take_profit_met(prices, entry_index, current_index, is_buy, lookback, min_profit):
    if is_buy:
        window = prices[entry_index:current_index + 1]
        if len(window) >= lookback:
            max_in_window = max(window[-lookback:])
            profit = max_in_window - prices[entry_index]
            return profit >= min_profit, profit
    else:
        window = prices[entry_index:current_index + 1]
        if len(window) >= lookback:
            min_in_window = min(window[-lookback:])
            profit = prices[entry_index] - min_in_window
            return profit >= min_profit, profit
    return False, 0.0

def plot_entry_points_chart(df, logs):
    plt.figure(figsize=(12, 6))
    plt.title("Entry Points Chart")

    plt.plot(df['time'], df['close'], label='Price', linewidth=0.8)

    buy_x = [log['date'] for log in logs if log['type'] == 'BUY']
    buy_y = [log['entry_price'] for log in logs if log['type'] == 'BUY']
    sell_x = [log['date'] for log in logs if log['type'] == 'SELL']
    sell_y = [log['entry_price'] for log in logs if log['type'] == 'SELL']
    tp_x = [log['date'] for log in logs if log['type'] == 'TP']
    tp_y = [log['entry_price'] for log in logs if log['type'] == 'TP']

    plt.scatter(buy_x, buy_y, color='green', label='BUY', marker='^')
    plt.scatter(sell_x, sell_y, color='red', label='SELL', marker='v')
    plt.scatter(tp_x, tp_y, facecolors='none', edgecolors='blue', label='TP', marker='o')

    plt.legend()
    plt.grid()
    plt.show()

def run_backtest(df, predicted_close, period_days=3, lookback=3, min_profit_pips=0.1):
    # 新機能：ポジション管理リセット
    ResetPositionState()

    total_trades = 0
    win_trades = 0
    loss_trades = 0
    total_profit = 0.0
    total_profit_amount = 0.0
    total_loss_amount = 0.0
    max_drawdown = 0.0
    entry_logs = []
    tp_success_count = 0  # TP成立回数カウンタ

    df = df.copy().reset_index(drop=False)

    for i in range(len(df) - period_days):
        # 🔒 保有中ならスキップ（多重エントリー防止）
        if IsPositionActive(i):
            continue

        rsi         = df.loc[i, "RSI_14"]
        macd        = df.loc[i, "MACD"]
        macd_signal = df.loc[i, "MACD_signal"]
        close_entry = df.loc[i, "close"]

        local_pred = predicted_close[i:i+5]
        if len(local_pred) < 5 or any(p is None for p in local_pred):
            continue

        pred_up = sum([local_pred[j] < local_pred[j+1] for j in range(len(local_pred)-1)])
        pred_down = sum([local_pred[j] > local_pred[j+1] for j in range(len(local_pred)-1)])

        entry_index = i

        # BUY Signal
        if pred_up >= 3 and macd > macd_signal and rsi < 50:
            total_trades += 1
            success = False


            # ✅ 保有状態に設定（ここがSetのタイミング）
            SetPositionActive(period_days, i)

            for j in range(1, period_days + 1):
                current_index = i + j
                tp_met, profit = is_take_profit_met(df["close"], i, current_index, is_buy=True,
                                                    lookback=lookback, min_profit=min_profit_pips)
                drawdown = close_entry - df.loc[current_index, "close"]
                max_drawdown = max(max_drawdown, drawdown)

                if tp_met:
                    win_trades += 1
                    total_profit += profit
                    total_profit_amount += profit
                    entry_logs.append({
                        "date": df.loc[current_index, "time"],
                        "entry_price": df.loc[current_index, "close"],
                        "type": "TP"
                    })
                    tp_success_count += 1
                    success = True

                    # TP成立時のみその場でクローズ
                    ClosePosition()

                    break

            if not success:
                profit = df.loc[i + period_days, "close"] - close_entry
                total_profit += profit
                if profit > 0:
                    total_profit_amount += profit
                    win_trades += 1
                else:
                    total_loss_amount += abs(profit)
                    loss_trades += 1
                # ✅ 満了時はここで明示的にポジションクローズ
                ClosePosition()

            entry_logs.append({
                "date": df.loc[entry_index, "time"],
                "entry_price": close_entry,
                "type": "BUY",
                "profit": profit,
                "success": success
            })

        # SELL Signal
        elif pred_down >= 3 and macd < macd_signal and rsi > 50:
            total_trades += 1
            success = False

            # ✅ 保有状態に設定（SELLでも同様）
            SetPositionActive(period_days, i)

            for j in range(1, period_days + 1):
                current_index = i + j
                tp_met, profit = is_take_profit_met(df["close"], i, current_index, is_buy=False,
                                                    lookback=lookback, min_profit=min_profit_pips)
                drawdown = df.loc[current_index, "close"] - close_entry
                max_drawdown = max(max_drawdown, drawdown)

                if tp_met:
                    win_trades += 1
                    total_profit += profit
                    total_profit_amount += profit
                    entry_logs.append({
                        "date": df.loc[current_index, "time"],
                        "entry_price": df.loc[current_index, "close"],
                        "type": "TP"
                    })
                    tp_success_count += 1
                    success = True

                    # TP成立時のみその場でクローズ
                    ClosePosition()

                    break

            if not success:
                profit = close_entry - df.loc[i + period_days, "close"]
                total_profit += profit
                
                if profit > 0:
                    total_profit_amount += profit
                    win_trades += 1
                else:
                    total_loss_amount += abs(profit)
                    loss_trades += 1

                # ✅ 満了時はここで明示的にポジションクローズ
                ClosePosition()

            entry_logs.append({
                "date": df.loc[entry_index, "time"],
                "entry_price": close_entry,
                "type": "SELL",
                "profit": profit,
                "success": success
            })

    profit_factor = (total_profit_amount / total_loss_amount) if total_loss_amount > 0 else float("inf")

    result = {
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": (win_trades / total_trades) * 100 if total_trades > 0 else 0,
        "average_profit": (total_profit / total_trades) if total_trades > 0 else 0,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "entry_logs": entry_logs
    }

    print(f"[DEBUG] TP condition met count: {tp_success_count}")
    if total_trades > 0:
        print(f"[DEBUG] TP hit rate: {(tp_success_count / total_trades) * 100:.2f}%")
    if len(entry_logs) > 0:
        plot_entry_points_chart(df, entry_logs)

    return result
