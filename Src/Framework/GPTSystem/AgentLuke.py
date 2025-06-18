
# 追加するディレクトリを指定
import sys
import os
new_path = os.path.join(os.getenv('PYTHON_ROOT'),"site-packages")
# sys.pathに新しいパスを追加
sys.path.append(new_path)

import openai

def AgentInitialize():
    """
    ルークとの連携準備
    """
    # 関数で行う処理
    # APIキーの設定
    #openai.api_key = "APIキー" 