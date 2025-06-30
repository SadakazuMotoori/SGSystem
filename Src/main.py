import numpy                                as np
import time
import keyboard

from Framework.MTSystem.MTManager           import MTManager_Initialize , MTManager_UpdateIndicators , draw_chart_with_trend_labels
from Framework.ForecastSystem.LSTMModel     import train_and_predict_lstm
from Framework.ForecastSystem.LSTMModel     import generate_predicted_series
from Framework.ForecastSystem.SignalEngine  import apply_trend_labels , PhaseA_Filter , show_trend_labels_in_period

from Framework.Utility.BacktestEngine       import run_backtest
from Framework.Utility.Utility              import NotificationManager

def main():
    _doBackTest = True
    print("==========SGSystem Start==========")
    
    # MT5初期化
    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。終了します。")
        quit()

    # 実践実行
    if not _doBackTest:
        print("[INFO] インジケータ更新と学習開始")
        df = MTManager_UpdateIndicators()

        # Phase-A: 環境認識＋ボラ判定
        if not PhaseA_Filter(df):
            print("[INFO] Phase-A 不通過：環境またはボラティリティ条件が一致せず")
            return  # 以降の処理は中断

        print("[INFO] Phase-A 通過：次フェーズへ進行")

        # Phase-A 通過後、学習と予測へ
        # LSTMによる予測処理（翌日の終値）
        predicted_prices = train_and_predict_lstm(df,True)
        # 例: 過去90日を使って予測 → 予測値は末尾だけ
        #predicted_prices は一部期間分しか無いので、NaN埋めで全体長に合わせる
        df["predicted_close"] = np.nan
        df.loc[df.index[-len(predicted_prices):], "predicted_close"] = predicted_prices

        print("[INFO] Phase-B シグナル判定")
#        signal = PhaseB_Trigger(predicted_prices, df)
        print(f"[SIGNAL] Phase-B 判定結果: {signal}")
        # TODO: signal をもとに売買処理 or 次フェーズへ

        # ESCキーで終了待ち
        while True:
            print('[INFO] 処理待機中（ESCキーで終了）...')
            if keyboard.is_pressed('escape'):
                break
            time.sleep(1)

    # バックテスト実行
    else:
        # ① インジケータ取得（※描画はまだ行わないようにする）
        df = MTManager_UpdateIndicators()

        # ② トレンドラベル付与
        df = apply_trend_labels(df, period=60, slope_threshold=0.005, adx_threshold=20, verbose=False)

        # ③ チャート描画（トレンドラベル含む）
#        df_zoom = df.loc["2023-06-01":"2023-10-15"]
#        df_zoom = df.loc["2024-08-01":"2024-11-15"]
        draw_chart_with_trend_labels(df)
        print("[DEBUG] ラベル集計")
        print(df["Trend_Label"].value_counts())
        print(df[df["Trend_Label"] != "no_trend"].tail(10))


        # ④ ログ確認
        print(df[["close", "Trend_Label"]].tail(10))
        show_trend_labels_in_period(df, "2023-06-01", "2023-11-01")
        show_trend_labels_in_period(df, "2024-08-01", "2024-11-15")

#        notifier = NotificationManager()
#        notifier.send_email("【SGSystem通知】テストメール","ルークより：これはGmailによる自動通知テストだ。")

#        print("[INFO] バックテスト開始中...")
#        print("[INFO] インジケータ更新と学習開始")
#        df = MTManager_UpdateIndicators()
        
#        df = apply_trend_labels(df, period=60, slope_threshold=0.005, adx_threshold=20, verbose=False)
        # ③ 判定結果の確認（例：末尾だけ表示）
#        print(df[["close", "Trend_Label"]].tail(10))

        # 2023年夏～秋の上昇トレンド
#        show_trend_labels_in_period(df, "2023-06-01", "2023-11-01")

        # 2024年秋の下降トレンド
#        show_trend_labels_in_period(df, "2024-08-01", "2024-11-15")

        #trend_label = PhaseA_Filter(df)

        # LSTMによる予測処理
 #       predicted_series = generate_predicted_series(df)
        # バックテスト用：全期間の逐次予測値を生成
 #       backtest_result = run_backtest(df, predicted_series)
    
 #       print("\n[Backtest Result]")
 #       print(f"Total Trades    : {backtest_result['total_trades']}")
 #       print(f"Win Trades      : {backtest_result['win_trades']}")
 #       print(f"Loss Trades     : {backtest_result['loss_trades']}")
 #       print(f"Win Rate        : {backtest_result['win_rate']:.2f}%")
 #       print(f"Average Profit  : {backtest_result['average_profit']:.3f} pips")
 #       print(f"Max Drawdown    : {backtest_result['max_drawdown']:.3f} pips")

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()
