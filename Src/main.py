#import sys
import time

# 追加するディレクトリを指定
#import os

#new_path = os.path.join(os.getenv('PYTHON_ROOT'),"site-packages")
# sys.pathに新しいパスを追加
#sys.path.append(new_path)
from Framework.GPTSystem.AgentLuke import AgentInitialize
from Framework.MTSystem.MTManager import MTManagerInitialize

import keyboard

def main():
    # 各システムの初期化
    print("==========SGSystem Start==========")

    # GPTSystemの初期化
    AgentInitialize()

    # MetaTraderSystemの初期化
    MTManagerInitialize()

    while True:
        print('processing...')
        
        # ESCキーが押されたかチェック
        if keyboard.is_pressed('escape'):
            break

        time.sleep(1)

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()