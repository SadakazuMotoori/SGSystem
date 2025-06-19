import sys
import os
import openai

# OPEN AI API_KEYを取得
api_key = os.getenv("OPENAI_API_KEY")

def AgentInitialize():
    """
    ルークとの連携準備
    """
    print("Agent Initialize")

    # パーソナルデータの構築
    #with open("lukes_personality_20250619.txt", "r", encoding="utf-8") as f:
    #system_prompt = f.read()

    #messages = [
    #    {"role": "system", "content": system_prompt},
    #    {"role": "user", "content": "現在のドル円相場の見通しは？"}
    #]

    #response = openai.ChatCompletion.create(
    #    model="gpt-4",
    #    messages=messages
    #)

    # 関数で行う処理
    # APIキーの設定
    #openai.api_key = "APIキー" 