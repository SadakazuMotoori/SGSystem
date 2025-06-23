import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

def is_buy_signal(predicted_close, rsi, macd, macd_signal):
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

def run_backtest(df, predicted_close, period_days=3, rsi_exit_threshold=70):
    """
    Phase-B ロジックに準拠したバックテストを実行
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

        if is_buy_signal(predicted_close, rsi, macd, macd_signal):
            total_trades += 1
            entry_price = close_entry
            entry_index = i
            success = False

            for j in range(1, period_days + 1):
                future_rsi = df.loc[i + j, "RSI_14"]
                future_price = df.loc[i + j, "close"]
                drawdown = entry_price - future_price
                max_drawdown = max(max_drawdown, drawdown)

                if future_rsi > rsi_exit_threshold:
                    profit = future_price - entry_price
                    total_profit += profit
                    win_trades += 1
                    success = True
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
    buys = [log for log in entry_logs if log["type"] == "BUY"]  # BUYログだけ抽出
    buy_df = pd.DataFrame({
        "date": pd.to_datetime([log["date"] for log in buys]),      # 日時（datetime型）
        "price": [log["entry_price"] for log in buys]               # エントリー価格
    })
    buy_df = buy_df[buy_df["date"].isin(ohlc.index)]  # OHLCに存在する日付のみ許可
    x_buys = [ohlc.index.get_loc(date) for date in buy_df["date"]]  # index位置取得（x軸）
    y_buys = buy_df["price"].values                                   # y軸は価格
    print(f"[DEBUG] BUY marker count after filtering: {len(buy_df)}")

    # === SELLエントリーの処理 ===
    sells = [log for log in entry_logs if log["type"] == "SELL"]  # SELLログだけ抽出
    sell_df = pd.DataFrame({
        "date": pd.to_datetime([log["date"] for log in sells]),     # 日時（datetime型）
        "price": [log["entry_price"] for log in sells]              # エントリー価格
    })
    sell_df = sell_df[sell_df["date"].isin(ohlc.index)]  # OHLCに存在する日付のみ許可
    x_sells = [ohlc.index.get_loc(date) for date in sell_df["date"]]  # index位置（x軸）
    y_sells = sell_df["price"].values                                  # y軸は価格
    print(f"[DEBUG] SELL marker count after filtering: {len(sell_df)}")

    # === データ整合性チェック ===
    if len(x_buys) != len(y_buys) or len(x_sells) != len(y_sells):
        print("[ERROR] XとYの長さ不一致：描画中止")
        return

    # === 表示対象が1件も無い場合の処理 ===
    if len(x_buys) == 0 and len(x_sells) == 0:
        print("[INFO] BUY/SELLエントリーが存在しません。チャート描画スキップ。")
        return

    # --- チャート描画本体（mplfinance） ---
    fig, axlist = mpf.plot(
        ohlc,
        type="candle",               # ローソク足
        style="yahoo",               # スタイル
        title="Entry Points Chart",  # タイトル
        ylabel="Price",              # Y軸ラベル
        volume=True,                 # 出来高表示
        returnfig=True,              # matplotlibオブジェクトを返す
        warn_too_much_data=len(ohlc)+1  # 警告抑制のためデータ数指定
    )

    # --- マーカー描画（BUY / SELL） ---
    ax_price = axlist[0]  # チャート本体の描画領域

    if len(x_buys) > 0:
        ax_price.scatter(
            x_buys, y_buys,
            marker='^', color='green', s=100, label='BUY'
        )

    if len(x_sells) > 0:
        ax_price.scatter(
            x_sells, y_sells,
            marker='v', color='red', s=100, label='SELL'
        )

    # --- 凡例表示 & 描画実行 ---
    ax_price.legend()
    plt.show()






