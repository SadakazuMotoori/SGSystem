import time
import keyboard

from Framework.MTSystem.MTManager import MTManager_Initialize, MTManager_UpdateIndicators
from Framework.ForecastSystem.LSTMModel import train_and_predict_lstm

def main():
    print("==========SGSystem Start==========")

    # MT5初期化
    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。終了します。")
        quit()

    print("[INFO] インジケータ更新と学習開始")
    df = MTManager_UpdateIndicators()

    # LSTMによる予測処理（翌日の終値）
    train_and_predict_lstm(df)

    # ESCキーで終了待ち
    while True:
        print('[INFO] 処理待機中（ESCキーで終了）...')
        if keyboard.is_pressed('escape'):
            break
        time.sleep(1)

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()
