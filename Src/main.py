import numpy                                as np
import time
import keyboard

from Framework.MTSystem.MTManager           import MTManager_Initialize , MTManager_UpdateIndicators , draw_chart_with_trend_labels
from Framework.ForecastSystem.LSTMModel     import train_and_predict_lstm
from Framework.ForecastSystem.SignalEngine  import PhaseA_Filter , show_trend_labels_in_period

from Framework.Utility.Utility              import NotificationManager

def main():
    _isTest = True
    print("==========SGSystem Start==========")
    
    # MT5初期化
    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。終了します。")
        quit()

    # 実践実行
    if not _isTest:
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
#        notifier = NotificationManager()
#        notifier.send_email("【SGSystem通知】テストメール","ルークより：これはGmailによる自動通知テストだ。")

        _enableTrade = True
        # ===================================================
        # PhaseA（トレンド確認）
        # ===================================================
        if _enableTrade:
            # ① インジケータ取得（ついでにトレンド情報も取得）
            df, trend_signal = MTManager_UpdateIndicators()

            if (trend_signal == "uptrend") or (trend_signal == "downtrend"):
                # 買い候補/売り候補としてLSTMへ
                print("[INFO] シグナル発生 = ",trend_signal)
            else:
                print("[INFO] トレンドシグナルなし → LSTMスキップ")
                _enableTrade = False

        # ③ チャート描画（トレンドラベル含む＆LSTM予測表示）
#        df_zoom = df.loc["2023-06-01":"2023-10-15"]
#        df = df.loc["2024-08-01":"2024-11-15"]
        draw_chart_with_trend_labels(df)

        # ===================================================
        # PhaseB（LSTMモデル実行：予測）
        # ===================================================
        if not _enableTrade:
            df = train_and_predict_lstm(df,True)

        

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()
