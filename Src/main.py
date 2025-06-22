import time
import keyboard

from Framework.MTSystem.MTManager           import MTManager_Initialize, MTManager_UpdateIndicators
from Framework.ForecastSystem.LSTMModel     import train_and_predict_lstm
from Framework.ForecastSystem.SignalEngine  import PhaseA_Filter, PhaseB_Trigger

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
