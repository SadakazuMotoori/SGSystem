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
            
            # --- Step 1: 形成中ローソク足を取り出して保存
            forming_candle = df.iloc[-1].copy()

            # --- Step 2: 未確定足を除外
            df = df.iloc[:-1]

            # --- Step 3: LSTM予測
            predicted_prices, df = LSTMModel_PredictLSTM(df, False)

            # --- Step 4: 形成中ローソク足を復元（次の日付で）
            forming_date = df.index[-1] + pd.Timedelta(days=1)
            while forming_date in df.index:
                forming_date += pd.Timedelta(days=1)
            df.loc[forming_date] = forming_candle

            # --- Step 5: さらにその翌日から予測追加
            for i, price in enumerate(predicted_prices):
                future_date = forming_date + pd.Timedelta(days=i + 1)
                while future_date in df.index:
                    future_date += pd.Timedelta(days=1)
                df.loc[future_date] = np.nan
                df.at[future_date, "LSTM_Predicted"] = price

            MTManager_DrawChart(df)


            # ============ 通知処理 ============
            latest_rsi = df["RSI_14"].dropna().iloc[-2]
            support = df["Support"].iloc[-2]
            resistance = df["Resistance"].iloc[-2]

            alerter.check_rsi_alert(latest_rsi)
            alerter.check_prediction_alert(predicted_prices[0], support, resistance)

            subject = f"【SGSystem予測】{df.index[-2].date()}時点"
            body = f""" ■ トレンドシグナル：{trend_signal}
                        ■ LSTM予測終値：[{predicted_prices[0]:.2f}, {predicted_prices[1]:.2f}, {predicted_prices[2]:.2f}, {predicted_prices[3]:.2f}, {predicted_prices[4]:.2f}]
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
            
            # --- Step 1: 形成中ローソク足を取り出して保存
            forming_candle = df.iloc[-1].copy()

            # --- Step 2: 未確定足を除外
            df = df.iloc[:-1]

            # --- Step 3: LSTM予測
            predicted_prices, df = LSTMModel_PredictLSTM(df, False)

            # --- Step 4: 形成中ローソク足を復元（次の日付で）
            forming_date = df.index[-1] + pd.Timedelta(days=1)
            while forming_date in df.index:
                forming_date += pd.Timedelta(days=1)
            df.loc[forming_date] = forming_candle

            # --- Step 5: さらにその翌日から予測追加
            for i, price in enumerate(predicted_prices):
                future_date = forming_date + pd.Timedelta(days=i + 1)
                while future_date in df.index:
                    future_date += pd.Timedelta(days=1)
                df.loc[future_date] = np.nan
                df.at[future_date, "LSTM_Predicted"] = price

            MTManager_DrawChart(df)


            # ============ 通知処理 ============
            latest_rsi = df["RSI_14"].dropna().iloc[-2]
            support = df["Support"].iloc[-2]
            resistance = df["Resistance"].iloc[-2]

            alerter.check_rsi_alert(latest_rsi)
            alerter.check_prediction_alert(predicted_prices[0], support, resistance)

            subject = f"【SGSystem予測】{df.index[-2].date()}時点"
            body = f""" ■ トレンドシグナル：{trend_signal or 'No Signal'}
                        ■ LSTM予測終値：[{predicted_prices[0]:.2f}, {predicted_prices[1]:.2f}, {predicted_prices[2]:.2f}, {predicted_prices[3]:.2f}, {predicted_prices[4]:.2f}]
                        ■ RSI：{latest_rsi:.2f}
                        ■ サポートライン：{support:.2f}
                        ■ レジスタンスライン：{resistance:.2f}
                        （チャート画像2枚を添付）"""

            notifier.send_email(subject, body, attachments=["Asset/Log/ChartImage/chart_full.png", "Asset/Log/ChartImage/chart_zoom.png"])

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()
