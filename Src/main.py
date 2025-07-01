from Framework.MTSystem.MTManager           import MTManager_Initialize , MTManager_UpdateIndicators , MTManager_DrawChart
from Framework.ForecastSystem.LSTMModel     import LSTMModel_PredictLSTM

from Framework.Utility.Utility              import NotificationManager

def main():
    _enableActual   = False
    _enableTrade    = True
    print("==========SGSystem Start==========")

    # ①MT5初期化
    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。終了します。")
        _enableTrade = False
        quit()

#   notifier = NotificationManager()
#   notifier.send_email("【SGSystem通知】テストメール","ルークより：これはGmailによる自動通知テストだ。")

    # ===================================================
    # 実戦実行
    # ===================================================
    if _enableActual:
        # ===================================================
        # ②PhaseA（トレンド確認）
        # ===================================================
        if _enableTrade:
            # インジケータ取得（ついでにトレンド情報も取得）
            df, trend_signal = MTManager_UpdateIndicators()

            # シグナル発生：買い候補/売り候補としてLSTMへ
            if (trend_signal == "uptrend") or (trend_signal == "downtrend"):
                print("[INFO] シグナル発生 = ",trend_signal)

                # ===================================================
                # ③PhaseB（LSTMモデル実行：翌日の値を予測）
                # ===================================================
                df = LSTMModel_PredictLSTM(df,True)

            else:
                print("[INFO] トレンドシグナルなし → LSTMスキップ")
                _enableTrade = False

            # ===================================================
            # ④チャート描画（トレンドラベル含む）
            # ===================================================
            # df_zoom = df.loc["2023-06-01":"2023-10-15"]
            # df = df.loc["2024-08-01":"2024-11-15"]
            MTManager_DrawChart(df)

    # ===================================================
    # 検証実行
    # ===================================================
    else:
        if _enableTrade:
            # インジケータ取得（ついでにトレンド情報も取得）
            df, trend_signal = MTManager_UpdateIndicators()

            # LSTMモデル実行：予測
            df = LSTMModel_PredictLSTM(df,True)

            # チャート描画（トレンドラベル含む）
            # df = df.loc["2023-06-01":"2023-10-15"]
            # df = df.loc["2024-08-01":"2024-11-15"]
            MTManager_DrawChart(df)

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()
