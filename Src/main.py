from Framework.MTSystem.MTManager           import MTManager_Initialize , MTManager_UpdateIndicators , MTManager_DrawChart
from Framework.ForecastSystem.LSTMModel     import LSTMModel_PredictLSTM

from Framework.Utility.Utility              import NotificationManager
from Framework.Utility.Utility              import AlertManager
import pandas as pd
import numpy as np

def main():
    _enableActual   = False
    _enableTrade    = True
    print("==========SGSystem Start==========")

    notifier    = NotificationManager()
    alerter     = AlertManager()

    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。終了します。")
        _enableTrade = False
        quit()

    if _enableActual:
        if _enableTrade:
            df, trend_signal = MTManager_UpdateIndicators()
            if (trend_signal == "uptrend") or (trend_signal == "downtrend"):
                print("[INFO] シグナル発生 = ", trend_signal)
                predicted_price, df = LSTMModel_PredictLSTM(df, False)

                next_date = df.index[-1] + pd.Timedelta(days=1)
                df.loc[next_date] = df.iloc[-1]
                df["LSTM_Predicted"] = np.nan
                df.at[next_date, "LSTM_Predicted"] = predicted_price

                MTManager_DrawChart(df)

                # ============ 通知処理 ============
                latest_rsi = df["RSI_14"].dropna().iloc[-2]
                support = df["Support"].iloc[-2]
                resistance = df["Resistance"].iloc[-2]

                alerter.check_rsi_alert(latest_rsi)
                alerter.check_prediction_alert(predicted_price, support, resistance)

                subject = f"【SGSystem予測】{df.index[-2].date()}時点"
                body = f""" ■ トレンドシグナル：{trend_signal}
                            ■ LSTM予測終値：{predicted_price:.2f}
                            ■ RSI：{latest_rsi:.2f}
                            ■ サポートライン：{support:.2f}
                            ■ レジスタンスライン：{resistance:.2f}
                            （チャート画像2枚を添付）"""
                
                notifier.send_email(subject, body, attachments=["Asset/Log/ChartImage/chart_full.png", "Asset/Log/ChartImage/chart_zoom.png"])
            else:
                print("[INFO] トレンドシグナルなし → LSTMスキップ")
                _enableTrade = False
    else:
        if _enableTrade:
            df, trend_signal = MTManager_UpdateIndicators()
            predicted_price, df = LSTMModel_PredictLSTM(df, False)

            next_date = df.index[-1] + pd.Timedelta(days=1)
            df.loc[next_date] = df.iloc[-1]
            df["LSTM_Predicted"] = np.nan
            df.at[next_date, "LSTM_Predicted"] = predicted_price

            MTManager_DrawChart(df)

            # ============ 通知処理 ============
            latest_rsi = df["RSI_14"].dropna().iloc[-2]
            support = df["Support"].iloc[-2]
            resistance = df["Resistance"].iloc[-2]

            alerter.check_rsi_alert(latest_rsi)
            alerter.check_prediction_alert(predicted_price, support, resistance)

            subject = f"【SGSystem予測】{df.index[-2].date()}時点"
            body = f""" ■ トレンドシグナル：{trend_signal or 'No Signal'}
                        ■ LSTM予測終値：{predicted_price:.2f}
                        ■ RSI：{latest_rsi:.2f}
                        ■ サポートライン：{support:.2f}
                        ■ レジスタンスライン：{resistance:.2f}
                        （チャート画像2枚を添付）"""

            notifier.send_email(subject, body, attachments=["Asset/Log/ChartImage/chart_full.png", "Asset/Log/ChartImage/chart_zoom.png"])

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()
