import sys
import time



# 追加するディレクトリを指定
import os
new_path = os.path.join(os.getenv('PYTHON_ROOT'),"site-packages")
# sys.pathに新しいパスを追加
sys.path.append(new_path)
from Framework.GPTSystem.AgentLuke import AgentInitialize

import keyboard

def main():
    # システムの初期化
    print("==========SGSystem Start==========")

    # GPTSystemの初期化
    #AgentInitialize()

    while True:
        print('processing...')
        
        # ESCキーが押されたかチェック
        if keyboard.is_pressed('escape'):
            break

        time.sleep(1)

    print("==========SGSystem End==========")

if __name__ == "__main__":
    main()