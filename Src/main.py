import numpy                                as np
import time
import keyboard

from Framework.MTSystem.MTManager           import MTManager_Initialize, MTManager_UpdateIndicators
from Framework.ForecastSystem.LSTMModel     import train_and_predict_lstm
from Framework.ForecastSystem.SignalEngine  import PhaseA_Filter, PhaseB_Trigger

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
        signal = PhaseB_Trigger(predicted_prices, df)
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
        NotificationManager.send_email("","","")

        print("[INFO] バックテスト開始中...")
        print("[INFO] インジケータ更新と学習開始")
        df = MTManager_UpdateIndicators()
        # LSTMによる予測処理（翌日の終値）
        predicted_prices = train_and_predict_lstm(df,True)
        # バックテスト用：全期間の逐次予測値を生成
        backtest_result = run_backtest(df, predicted_prices)
    
        print("\n[Backtest Result]")
        print(f"Total Trades    : {backtest_result['total_trades']}")
        print(f"Win Trades      : {backtest_result['win_trades']}")
        print(f"Loss Trades     : {backtest_result['loss_trades']}")
        print(f"Win Rate        : {backtest_result['win_rate']:.2f}%")
        print(f"Average Profit  : {backtest_result['average_profit']:.3f} pips")
        print(f"Max Drawdown    : {backtest_result['max_drawdown']:.3f} pips")

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()
