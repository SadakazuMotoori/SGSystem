import time
import keyboard

from Framework.MTSystem.MTManager           import MTManager_Initialize, MTManager_UpdateIndicators
from Framework.ForecastSystem.LSTMModel     import train_and_predict_lstm
from Framework.ForecastSystem.SignalEngine  import PhaseA_Filter, PhaseB_Trigger

from Framework.Utility.BacktestEngine       import run_backtest

def main():
    print("==========SGSystem Start==========")

    # MT5初期化
    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。終了します。")
        quit()

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

    # ======== バックテスト開始 ========
    print("[INFO] バックテスト開始中...")
    backtest_result = run_backtest(df)
    
    print("\n[Backtest Result]")
    print(f"Total Trades    : {backtest_result['total_trades']}")
    print(f"Win Trades      : {backtest_result['win_trades']}")
    print(f"Loss Trades     : {backtest_result['loss_trades']}")
    print(f"Win Rate        : {backtest_result['win_rate']:.2f}%")
    print(f"Average Profit  : {backtest_result['average_profit']:.3f} pips")
    print(f"Max Drawdown    : {backtest_result['max_drawdown']:.3f} pips")
    # ======== バックテスト終了 ========

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

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()
