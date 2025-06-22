
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

def run_backtest(df: pd.DataFrame, holding_days: int = 3, rsi_threshold: float = 70.0, period_days: int = 60, plot=True):
    print("\n========== Backtest Start ==========")

    df = df.dropna().copy()
    df = df.iloc[-period_days:].copy()

    total_trades = 0
    win_trades = 0
    loss_trades = 0
    profit_list = []

    # エントリーポイント記録用（可視化用）
    entry_indices = []
    entry_colors = []

    for i in range(len(df) - holding_days):
        entry_price = df.iloc[i]["close"]
        success = False
        exit_price = df.iloc[i + holding_days]["close"]

        # === [NEW] 予測終値と誤差のログ出力 ===
        if "predicted_close" in df.columns:
            predicted_close = df.iloc[i]["predicted_close"]
            actual_close = df.iloc[i + holding_days]["close"]
            error = predicted_close - actual_close
            print(f"[PREDICT] {df.index[i]} → Predict: {predicted_close:.3f} | Actual: {actual_close:.3f} | Error: {error:+.3f}")
        # ==================================

        # RSIが閾値を超えたらそこで利確
        for j in range(1, holding_days + 1):
            if df.iloc[i + j]["RSI_14"] >= rsi_threshold:
                exit_price = df.iloc[i + j]["close"]
                success = True
                break

        pnl = exit_price - entry_price
        profit_list.append(pnl)
        total_trades += 1

        entry_date = df.index[i]
        entry_indices.append(entry_date)
        if success:
            win_trades += 1
            entry_colors.append('green')
        else:
            loss_trades += 1
            entry_colors.append('red')

        print("[DEBUG] df.columns:", df.columns)
        print(f"[Entry] {entry_date.date()} Close={entry_price:.2f} RSI={df.iloc[i]['RSI_14']:.1f} -> Exit={exit_price:.2f} {'✅' if success else '❌'}")

    # チャート表示（1回だけ）
    if plot:
        print("[DEBUG] plot block entered")
        print("[DEBUG] df.index.head():", df.index[:5])
        print("[DEBUG] sample date_to_idx:", {date: i for i, date in enumerate(df.index[:5])})

        print("[INFO] Plotting chart now...")
        apds = [mpf.make_addplot(df["RSI_14"], panel=1, ylabel='RSI')]

        # 成功・失敗を分けたマーカー用データ
        success_data = []
        fail_data = []

        valid_entries = [
            (idx, color)
            for idx, color in zip(entry_indices, entry_colors)
            if idx in df.index
        ]

        for idx, color in valid_entries:
            price = df.loc[idx, "close"]
            if color == 'green':
                success_data.append((idx, price))
            elif color == 'red':
                fail_data.append((idx, price))

        print(f"[DEBUG] success: {len(success_data)} points")
        print(f"[DEBUG] fail: {len(fail_data)} points")

        # mpf.plot(): returnfig=True で Figure, Axes を取得
        fig, axes = mpf.plot(
            df,
            type='candle',
            style='yahoo',
            addplot=apds,
            volume=True,
            title="Backtest Entry Points",
            returnfig=True
        )

        ax_main = axes[0]  # ローソク足描画用のメインチャート

        # mplfinance の x 軸は整数インデックス。これと同じスケールにする必要あり。
        date_to_idx = {date: i for i, date in enumerate(df.index)}

        # 失敗データのみプロット（ここでエラーの元だった scatter を手動で処理）
        if fail_data:
            fail_dates, fail_prices = zip(*fail_data)
            fail_x = [date_to_idx[d] for d in fail_dates]
            ax_main.scatter(
                fail_x,
                fail_prices,
                marker='^',
                color='red',
                s=100,
                label='Failed Entry'
            )

        if success_data:
            success_dates, success_prices = zip(*success_data)
            success_x = [date_to_idx[d] for d in success_dates]
            ax_main.scatter(
                success_x,
                success_prices,
                marker='^',
                color='green',
                s=100,
                label='Successful Entry'
            )

        # plt.show() によって明示的に描画
        import matplotlib.pyplot as plt
        plt.show()

    win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_profit = np.mean(profit_list) if profit_list else 0
    max_drawdown = min(profit_list) if profit_list else 0

    print(f"総トレード数: {total_trades}")
    print(f"勝ちトレード数: {win_trades}")
    print(f"負けトレード数: {loss_trades}")
    print(f"勝率: {win_rate:.2f}%")
    print(f"平均損益: {avg_profit:.3f} pips")
    print(f"最大ドローダウン: {max_drawdown:.3f} pips")
    print("========== Backtest End ==========\n")

    plt.show()

    return {
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": win_rate,
        "average_profit": avg_profit,
        "max_drawdown": max_drawdown
    }
