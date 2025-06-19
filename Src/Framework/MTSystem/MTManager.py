import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def MTManagerInitialize():
    """
    MetaTraderシステムとの連携準備
    """
    print("MTManager Initialize")

    # MT5へ接続
    loginID     = int(os.getenv('MT_LOGIN_ID'))
    loginPass   = os.getenv('MT_LOGIN_PASS')
    if not mt5.initialize(login=loginID, server="OANDA-Japan MT5 Live",password=loginPass):
        print("接続失敗：", mt5.last_error())
        return False
    
    # シンボル設定（ブローカーの仕様による。必要に応じて調整）
    symbol = "USDJPY"

    # 過去100日分のD1ローソク足取得
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 100)

    # DataFrame化
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 表示
    print(df.tail())

    return True