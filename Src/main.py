from Framework.MTSystem.MTManager           import MTManager_Initialize , MTManager_UpdateIndicators , MTManager_DrawChart
from Framework.ForecastSystem.LSTMModel     import LSTMModel_PredictLSTM

from Framework.Utility.Utility              import NotificationManager
import pandas as pd
import numpy as np

def main():
    _enableActual   = False
    _enableTrade    = True
    print("==========SGSystem Start==========")

    # ①MT5初期化
    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。終了します。")
        _enableTrade = False
        quit()

    # ===================================================
    # 実戦実行
    # ===================================================
    if _enableActual:
        if _enableTrade:
            df, trend_signal = MTManager_UpdateIndicators()

            if (trend_signal == "uptrend") or (trend_signal == "downtrend"):
                print("[INFO] シグナル発生 = ",trend_signal)

                predicted_price, df = LSTMModel_PredictLSTM(df, False)

                # ▼ 翌日の行を追加し、予測値をプロット用に記録
                next_date = df.index[-1] + pd.Timedelta(days=1)
                df.loc[next_date] = df.iloc[-1]  # ダミー行（コピー）
                df["LSTM_Predicted"] = np.nan
                df.at[next_date, "LSTM_Predicted"] = predicted_price

            else:
                print("[INFO] トレンドシグナルなし → LSTMスキップ")
                _enableTrade = False

            MTManager_DrawChart(df)

    # ===================================================
    # 検証実行
    # ===================================================
    else:
        if _enableTrade:
            df, trend_signal = MTManager_UpdateIndicators()
            predicted_price, df = LSTMModel_PredictLSTM(df, False)

            next_date = df.index[-1] + pd.Timedelta(days=1)
            df.loc[next_date] = df.iloc[-1]
            df["LSTM_Predicted"] = np.nan
            df.at[next_date, "LSTM_Predicted"] = predicted_price

            MTManager_DrawChart(df)

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()
