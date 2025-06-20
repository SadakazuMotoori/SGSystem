import time
import keyboard

from Framework.GPTSystem.AgentLuke import AgentInitialize
from Framework.MTSystem.MTManager import MTManager_Initialize
from Framework.MTSystem.MTManager import MTManager_UpadteIndicators

def main():
    # 各システムの初期化
    print("==========SGSystem Start==========")

    # GPTSystemの初期化
 #   AgentInitialize()

    # MetaTraderSystemの初期化
    isSuccess =  MTManager_Initialize()
    if not isSuccess:
        quit()

    MTManager_UpadteIndicators()
    while True:
        print('processing...')
        
        # ESCキーが押されたかチェック
        if keyboard.is_pressed('escape'):
            break

        time.sleep(1)

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()