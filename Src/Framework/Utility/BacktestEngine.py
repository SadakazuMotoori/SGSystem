"""
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from Framework.ForecastSystem.SignalEngine import PhaseA_Filter, PhaseB_Trigger
from Framework.MTSystem.MTManager import (
    IsPositionActive,
    SetPositionActive,
    ResetPositionState,
    ClosePosition
)

def is_take_profit_met_trailing(prices, entry_index, current_index, is_buy, max_or_min_price, trailing_value):
    if is_buy:
        stop_price = max_or_min_price - trailing_value
        return prices[current_index] <= stop_price
    else:
        stop_price = max_or_min_price + trailing_value
        return prices[current_index] >= stop_price

def plot_entry_points_chart(df, logs):
    plt.figure(figsize=(12, 6))
    plt.title("Entry Points Chart")

    plt.plot(df['time'], df['close'], label='Price', linewidth=0.8)

    buy_x = [log['date'] for log in logs if log['type'] == 'BUY']
    buy_y = [log['entry_price'] for log in logs if log['type'] == 'BUY']
    sell_x = [log['date'] for log in logs if log['type'] == 'SELL']
    sell_y = [log['entry_price'] for log in logs if log['type'] == 'SELL']
    tp_x = [log['date'] for log in logs if log['type'] == 'TP']
    tp_y = [log['exit_price'] for log in logs if log['type'] == 'TP']

    plt.scatter(buy_x, buy_y, color='green', label='BUY', marker='^')
    plt.scatter(sell_x, sell_y, color='red', label='SELL', marker='v')
    plt.scatter(tp_x, tp_y, facecolors='none', edgecolors='blue', label='TP', marker='o')

    plt.legend()
    plt.grid()
    plt.show()

def run_backtest(df, predicted_close, period_days=5, lookback=3, use_trailing_tp=True):
    ResetPositionState()

    total_trades = 0
    win_trades = 0
    loss_trades = 0
    total_profit = 0.0
    total_profit_amount = 0.0
    total_loss_amount = 0.0
    max_drawdown = 0.0
    entry_logs = []
    tp_success_count = 0

    df = df.copy().reset_index(drop=False)

    for i in range(len(df) - period_days):
        if IsPositionActive(i):
            entry_logs.append({
                "date": df.loc[i, "time"],
                "entry_price": df.loc[i, "close"],
                "type": "SKIP",
                "reason": "ポジションが未決済のためスキップ"
            })
            continue

        current_df = df.iloc[:i+1].copy()
        pass_a, reason_a = PhaseA_Filter(current_df)
        if not pass_a:
            entry_logs.append({
                "date": df.loc[i, "time"],
                "entry_price": df.loc[i, "close"],
                "type": "SKIP",
                "reason": f"Phase-A失敗: {reason_a}"
            })
            continue

        current_pred = predicted_close[i:i+5]
        if len(current_pred) < 5 or any(p is None for p in current_pred):
            print(f"[SKIP] {df.loc[i, 'time']} - predicted_close不足: {current_pred}")
            continue

        signal, reason_b = PhaseB_Trigger(current_pred, current_df)
        if signal == "NO-TRADE":
            entry_logs.append({
                "date": df.loc[i, "time"],
                "entry_price": df.loc[i, "close"],
                "type": "SKIP",
                "reason": f"Phase-B不成立: {reason_b}"
            })
            continue

        entry_index = i
        close_entry = df.loc[i, "close"]

        is_buy = signal == "BUY"

        total_trades += 1
        SetPositionActive(period_days, i)
        success = False

        atr_now = df.loc[i, "ATR_14"]
        trailing_stop = atr_now * 0.8

        max_or_min_price = close_entry

        for j in range(1, period_days + 1):
            current_index = i + j
            current_price = df.loc[current_index, "close"]

            if is_buy:
                max_or_min_price = max(max_or_min_price, current_price)
                drawdown = close_entry - current_price
            else:
                max_or_min_price = min(max_or_min_price, current_price)
                drawdown = current_price - close_entry

            max_drawdown = max(max_drawdown, drawdown)

            if use_trailing_tp:
                tp_met = is_take_profit_met_trailing(
                    df["close"], i, current_index,
                    is_buy=is_buy,
                    max_or_min_price=max_or_min_price,
                    trailing_value=trailing_stop
                )
            else:
                tp_met = False

            if tp_met:
                win_trades += 1
                profit = current_price - close_entry if is_buy else close_entry - current_price
                total_profit += profit
                total_profit_amount += profit
                entry_logs.append({
                    "date": df.loc[current_index, "time"],
                    "entry_price": close_entry,
                    "exit_price": current_price,
                    "type": "TP",
                    "position": "BUY" if is_buy else "SELL",
                    "profit": profit,
                    "tp_type": "trailing"
                })
                tp_success_count += 1
                success = True
                ClosePosition()
                break

        if not success:
            final_price = df.loc[i + period_days, "close"]
            profit = final_price - close_entry if is_buy else close_entry - final_price
            total_profit += profit
            if profit > 0:
                win_trades += 1
                total_profit_amount += profit
            else:
                loss_trades += 1
                total_loss_amount += abs(profit)
            ClosePosition()

        entry_logs.append({
            "date": df.loc[entry_index, "time"],
            "entry_price": close_entry,
            "type": "BUY" if is_buy else "SELL",
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

    for log in entry_logs:
        if log["type"] == "SKIP":
            print(f"[SKIP] {log['date']}: {log['reason']}")

    print(f"[期間] {df['time'].iloc[0]} ～ {df['time'].iloc[-1]}")
    analyze_skips_by_period(result["entry_logs"], start_date=datetime(2023, 4, 1), end_date=datetime(2023, 7, 31))
    return result

def analyze_skips_by_period(entry_logs, start_date, end_date):
    print(f"[SKIP分析] {start_date.date()} ～ {end_date.date()} の除外理由一覧 ↓")
    for log in entry_logs:
        if log["type"] == "SKIP":
            log_date = pd.to_datetime(log["date"])
            if start_date <= log_date <= end_date:
                print(f"{log['date']} | {log['reason']}")
"""